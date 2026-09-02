#!/usr/bin/env python3
"""Materialize the deterministic Stage-D OI/funding lineage rejection audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/stage_d_oi_funding_lineage_audit_20260731_v4"
KRAKEN_ROOT = ROOT / "data_perp/exchanges/krakenfutures"
FUNDING_ZIP = KRAKEN_ROOT / "raw/funding_rates/kraken_historical_funding_rates.zip"
REFERENCE_ZIP = KRAKEN_ROOT / "reference/kraken_funding_rates_export_20260216.zip"
FUNDING_DIR = KRAKEN_ROOT / "funding_hourly"
OI_DIR = KRAKEN_ROOT / "open_interest_hourly"
LEGACY_SOURCE_DIRS = (
    ROOT / "data_perp/funding_hourly",
    ROOT / "data_perp/exchanges/binanceusdm/funding_hourly",
    ROOT / "data_perp/exchanges/binanceusdm/open_interest_hourly",
    ROOT / "data_perp/backups/oi_quote_unit_repair_20260716/open_interest_hourly",
    KRAKEN_ROOT / "oi_quote_unit_backup_20260716/open_interest_hourly",
)
ALLOWED_DISPOSITIONS = (
    "ADMITTED_CAUSAL",
    "REJECTED_NO_AVAILABILITY_TIMESTAMP",
    "REJECTED_UNBOUNDED_STALENESS",
    "REJECTED_PRODUCT_MISMATCH",
    "REJECTED_NO_LIVE_PARITY",
)
FORBIDDEN_FUTURE_FUNDING = (
    "next funding payment",
    "future funding estimate",
    "future settlement",
    "revised future value",
)
CODE_EVIDENCE = (
    ROOT / "extreme_price_movements/data_store.py",
    ROOT / "extreme_price_movements/inference/data_fetcher.py",
    ROOT / "scripts/import_kraken_historical_funding_rates.py",
    ROOT / "scripts/backfill_kraken_historical_funding_rates_api.py",
    ROOT / "scripts/backfill_kraken_open_interest_analytics.py",
    ROOT / "scripts/repair_kraken_open_interest_quote_units.py",
    ROOT / "scripts/replay_live_signal_predictions.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _directory_facts(directory: Path, value_column: str) -> dict[str, Any]:
    files = sorted(directory.glob("*.parquet"))
    rows = 0
    minimum = None
    maximum = None
    max_run = 0
    max_run_file = None
    columns: set[str] = set()
    for path in files:
        frame = pd.read_parquet(path)
        columns.update(map(str, frame.columns))
        rows += len(frame)
        index = frame.index if isinstance(frame.index, pd.DatetimeIndex) else frame.get("ts")
        timestamps = pd.to_datetime(index, utc=True, errors="coerce")
        if len(timestamps):
            current_min, current_max = timestamps.min(), timestamps.max()
            minimum = current_min if minimum is None else min(minimum, current_min)
            maximum = current_max if maximum is None else max(maximum, current_max)
        if value_column in frame and len(frame):
            values = pd.to_numeric(frame[value_column], errors="coerce")
            groups = (values.ne(values.shift()) | values.isna()).cumsum()
            run = int(values.groupby(groups).size().max())
            if run > max_run:
                max_run, max_run_file = run, path.name
    return {
        "files": len(files),
        "rows": rows,
        "start": minimum.isoformat() if minimum is not None else None,
        "end": maximum.isoformat() if maximum is not None else None,
        "columns": ",".join(sorted(columns)),
        "max_identical_run_hours": max_run,
        "max_identical_run_file": max_run_file,
    }


def _zip_facts(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name for name in archive.namelist() if name.startswith("exports/") and name.endswith(".csv")
        )
    pf = [name for name in members if Path(name).stem.startswith("PF_")]
    pi = [name for name in members if Path(name).stem.startswith("PI_")]
    underlying = lambda name: re.sub(r"^(PF|PI)_?", "", Path(name).stem)
    collisions = sorted(set(map(underlying, pf)) & set(map(underlying, pi)))
    return {"members": len(members), "pf": len(pf), "pi": len(pi), "collisions": collisions}


def _row(
    source_id: str,
    family: str,
    raw_source: str,
    provider: str,
    product_type: str,
    source_fields: str,
    source_interval: str,
    update_cadence: str,
    event_timestamp: str,
    observation_timestamp: str,
    availability_timestamp: str,
    ingestion_timestamp: str,
    settlement_semantics: str,
    publication_delay: str,
    revision_behavior: str,
    missingness: str,
    forward_fill_behavior: str,
    maximum_observed_staleness: str,
    live_inference_source: str,
    live_parity: bool,
    product_separated: bool,
    bounded_staleness: bool,
    future_funding_safe: bool,
    disposition: str,
    evidence: str,
) -> dict[str, Any]:
    return locals()


def build_ledger() -> pd.DataFrame:
    zip_facts = _zip_facts(FUNDING_ZIP)
    funding = _directory_facts(FUNDING_DIR, "funding_rate")
    oi = _directory_facts(OI_DIR, "open_interest")
    legacy_counts = {
        str(directory.relative_to(ROOT)): len(list(directory.glob("*.parquet")))
        for directory in LEGACY_SOURCE_DIRS
    }
    legacy_inventory = "; ".join(
        f"{path} ({count} parquet tables)" for path, count in legacy_counts.items()
    )
    collision_text = ",".join(zip_facts["collisions"])
    rows = [
        _row("funding_official_export_pf", "funding", str(FUNDING_ZIP.relative_to(ROOT)), "Kraken", "PF linear USD perpetual", "timestamp,tradeable,absolute_rate,relative_rate", "nominal hourly", "archive snapshot", "timestamp", "absent", "absent", "absent", "not retained", "unproven", "unrecorded", "symbol/history dependent", "none in raw archive", "unknowable", "historical export only", False, True, False, False, "REJECTED_NO_AVAILABILITY_TIMESTAMP", "import_kraken_historical_funding_rates.py:53-63; data_store.py:1039-1093; archive PF members=%d" % zip_facts["pf"]),
        _row("funding_reference_export_copy", "funding", str(REFERENCE_ZIP.relative_to(ROOT)), "Kraken", "mixed PF linear and PI inverse USD perpetual archive", "timestamp,tradeable,absolute_rate,relative_rate", "nominal hourly", "fixed reference archive snapshot", "timestamp", "absent", "absent", "absent", "not retained", "unproven", "unrecorded; byte-identical duplicate of raw archive", "symbol/history dependent", "none in raw archive", "unknowable", "reference archive, not a live feed", False, False, False, False, "REJECTED_NO_AVAILABILITY_TIMESTAMP", f"data_store.py:1044-1053; SHA256={sha256(REFERENCE_ZIP)}; byte_identical_to_raw={sha256(REFERENCE_ZIP) == sha256(FUNDING_ZIP)}; members={zip_facts['members']}"),
        _row("funding_official_export_pi", "funding", str(FUNDING_ZIP.relative_to(ROOT)), "Kraken", "PI inverse USD perpetual", "timestamp,tradeable,absolute_rate,relative_rate", "nominal hourly", "archive snapshot", "timestamp", "absent", "absent", "absent", "not retained", "unproven", "unrecorded", "symbol/history dependent", "none in raw archive", "unknowable", "frozen candidates require PF linear", False, False, False, False, "REJECTED_PRODUCT_MISMATCH", f"import_kraken_historical_funding_rates.py:38-46; PI members={zip_facts['pi']}; PF/PI collisions={collision_text}"),
        _row("funding_kraken_history_api", "funding", "https://futures.kraken.com/derivatives/api/v3/historical-funding-rates", "Kraken", "exchange market id; PF intended", "timestamp,relativeFundingRate|fundingRate", "hourly observations", "on-demand historical response", "timestamp", "absent", "absent", "not persisted", "not retained", "unproven", "API revisions merged by last", "endpoint/history dependent", "none before downstream persistence", "unknowable", "same endpoint callable live", True, True, False, False, "REJECTED_NO_AVAILABILITY_TIMESTAMP", "data_store.py:980-1036; backfill_kraken_historical_funding_rates_api.py:58-78"),
        _row("funding_ccxt_history_fallback", "funding", "CCXT fetch_funding_rate_history", "CCXT/Kraken", "resolved perpetual symbol", "timestamp|fundingTimestamp|time|fundingTime; fundingRate|rate", "provider dependent; treated hourly", "on-demand", "provider event timestamp floored hour", "absent", "absent", "not persisted", "not retained", "unproven", "group-last merge", "provider/history dependent", "none before downstream persistence", "unknowable", "callable in inference", True, True, False, False, "REJECTED_NO_AVAILABILITY_TIMESTAMP", "data_store.py:893-977,2897-2908,3871-3886; inference/data_fetcher.py:1667-1695"),
        _row("funding_live_ticker", "funding", "Kraken Futures publicGetTickers", "Kraken", "live exchange market id", "fundingRate", "snapshot", "10-second in-memory cache; caller driven", "caller timestamp floored hour", "absent", "absent", "not persisted exactly", "estimate/payment meaning not retained", "unproven", "overwritten by hourly group-last", "capture dependent", "merged with historical sidecar", "unknowable after persistence", "authoritative live ticker path", True, True, False, False, "REJECTED_NO_AVAILABILITY_TIMESTAMP", "inference/data_fetcher.py:504-580,1704-1765"),
        _row("funding_hourly_sidecars", "funding", str(FUNDING_DIR.relative_to(ROOT)), "mixed Kraken export/API/live", "PF/PI identity not persisted", funding["columns"], "hourly index", "mixed", "index only", "absent", "absent", "absent", "absent", "unproven", "source identity/revisions lost", "mixed", "unbounded reindex.ffill and embedded ffill", f"unbounded; max identical run {funding['max_identical_run_hours']}h in {funding['max_identical_run_file']}", "loaded by inference/replay", True, False, False, False, "REJECTED_UNBOUNDED_STALENESS", f"files={funding['files']}; rows={funding['rows']}; span={funding['start']}..{funding['end']}; replay_live_signal_predictions.py:1010-1014; data_store.py:3042,4029-4031,3504-3507"),
        _row("oi_kraken_analytics", "open_interest", "https://futures.kraken.com/api/charts/v1/analytics/{product}/open-interest", "Kraken", "exchange market id; PF intended", "chart time plus native contract/base amount", "requested 1h", "on-demand paged", "chart timestamp floored hour", "absent", "absent", "not persisted", "not applicable", "unproven", "group-last merge", "endpoint/history dependent", "none before downstream persistence", "unknowable", "endpoint callable during refresh", True, True, False, True, "REJECTED_NO_AVAILABILITY_TIMESTAMP", "data_store.py:1327-1409; native-to-quote conversion data_store.py:1412-1440"),
        _row("oi_ccxt_history_fallback", "open_interest", "CCXT fetch_open_interest_history", "CCXT/Kraken", "resolved perpetual symbol", "timestamp plus openInterest* fields", "provider dependent; treated hourly", "on-demand", "provider timestamp floored hour", "absent", "absent", "not persisted", "not applicable", "unproven", "group-last merge", "provider/history dependent", "none before downstream persistence", "unknowable", "callable in refresh", True, True, False, True, "REJECTED_NO_AVAILABILITY_TIMESTAMP", "data_store.py:893-977,3903-3926"),
        _row("oi_live_ticker", "open_interest", "Kraken Futures publicGetTickers", "Kraken", "live exchange market id", "openInterestValue or openInterest times mark/index", "snapshot", "10-second in-memory cache; caller driven", "caller timestamp floored hour", "absent", "absent", "not persisted exactly", "not applicable", "unproven", "overwritten by hourly group-last", "capture dependent", "merged with historical sidecar", "unknowable after persistence", "authoritative live ticker path", True, True, False, True, "REJECTED_NO_AVAILABILITY_TIMESTAMP", "inference/data_fetcher.py:504-580"),
        _row("oi_hourly_sidecars", "open_interest", str(OI_DIR.relative_to(ROOT)), "mixed Kraken analytics/embedded/live", "market id not persisted", oi["columns"], "hourly index", "mixed", "index only", "absent", "absent", "absent", "not applicable", "unproven", "source identity/revisions lost", "mixed", "unbounded replay ffill", f"unbounded; max identical run {oi['max_identical_run_hours']}h in {oi['max_identical_run_file']}", "loaded by inference/replay", True, False, False, True, "REJECTED_UNBOUNDED_STALENESS", f"files={oi['files']}; rows={oi['rows']}; span={oi['start']}..{oi['end']}; inference/data_fetcher.py:434-462; replay_live_signal_predictions.py:1010-1014; repair_kraken_open_interest_quote_units.py:2-8,122-168"),
        _row("embedded_hourly_ohlcv_aux", "open_interest+funding", "krakenfutures/ohlcv yearly partitions", "Kraken-derived mixed sources", "linear USD rows but upstream identity lost", "bar ts,funding_rate,open_interest", "hourly bar", "historical refresh/compaction", "OHLCV bar timestamp", "absent", "absent", "absent", "funding semantics absent", "unproven", "newer sparse values merged column-wise", "auxiliary dependent", "explicit unlimited ffill", "unbounded by implementation", "same store used live/replay", True, False, False, False, "REJECTED_UNBOUNDED_STALENESS", "data_store.py:3040-3043,3368-3424,3496-3507,4027-4034"),
        _row("legacy_unscoped_and_binance_sidecars", "open_interest+funding", legacy_inventory, "Binance/legacy unscoped and non-authoritative Kraken backups", "USDT/USDC, unknown, or archived Kraken copies; not an authoritative isolated Kraken linear-USD source", "funding_rate|open_interest,ts", "mixed", "legacy/archive", "ts", "absent", "absent", "absent", "provider dependent", "unproven", "unrecorded", "sparse/archive dependent", "consumer dependent", "unknowable", "canonical inference is exchange-scoped; backups are not consumed", False, False, False, False, "REJECTED_PRODUCT_MISMATCH", f"inference/data_fetcher.py:210-239; derived inventory: {legacy_inventory}. All files are hash-sealed; backup tables inherit rejected OI-sidecar lineage and are not authoritative inputs."),
    ]
    ledger = pd.DataFrame(rows)
    return ledger


def validate_ledger(ledger: pd.DataFrame) -> dict[str, Any]:
    failures: list[str] = []
    if len(ledger) != 13:
        failures.append(f"expected_13_source_classes_got_{len(ledger)}")
    invalid = sorted(set(ledger["disposition"]) - set(ALLOWED_DISPOSITIONS))
    if invalid:
        failures.append(f"invalid_dispositions={invalid}")
    admitted = ledger[ledger.disposition.eq("ADMITTED_CAUSAL")]
    if not admitted.empty:
        missing_availability = admitted[
            admitted["observation_timestamp"].eq("absent")
            | admitted["availability_timestamp"].eq("absent")
        ]
        if not missing_availability.empty:
            failures.append("admitted_source_missing_observation_or_availability_timestamp")
        if (~admitted["bounded_staleness"]).any():
            failures.append("admitted_source_has_unbounded_staleness")
        if (~admitted["product_separated"]).any():
            failures.append("admitted_source_lacks_product_separation")
        if (~admitted["live_parity"]).any():
            failures.append("admitted_source_lacks_live_parity")
    if ledger.loc[ledger.source_id.eq("funding_official_export_pi"), "disposition"].tolist() != ["REJECTED_PRODUCT_MISMATCH"]:
        failures.append("inverse_funding_product_not_rejected")
    funding = ledger[ledger.family.str.contains("funding", regex=False)]
    if funding["future_funding_safe"].any():
        failures.append("funding_source_incorrectly_marked_future_safe")
    if failures:
        raise ValueError("; ".join(failures))
    return {
        "passed": True,
        "source_classes": len(ledger),
        "admitted_sources": int(ledger.disposition.eq("ADMITTED_CAUSAL").sum()),
        "a6_disposition": "REJECTED_LINEAGE",
        "a7_disposition": "REJECTED_LINEAGE",
        "companion_disposition": "OI_FUNDING_CAUSAL_LINEAGE_UNRESOLVED",
        "checks": {
            "observation_and_availability_required": True,
            "unbounded_forward_fill_rejected": True,
            "pf_pi_product_separation_required": True,
            "live_parity_required": True,
            "future_funding_inputs_forbidden": list(FORBIDDEN_FUTURE_FUNDING),
        },
    }


def _report(ledger: pd.DataFrame, tests: dict[str, Any]) -> str:
    lines = [
        "# Stage-D OI/funding causal-lineage audit",
        "",
        "## Verdict",
        "",
        "`A6 = REJECTED_LINEAGE`  ",
        "`A7 = REJECTED_LINEAGE`  ",
        "`OI_FUNDING_CAUSAL_LINEAGE_UNRESOLVED`",
        "",
        "No historical source is `ADMITTED_CAUSAL`. This audit changes neither ingestion nor feature admission.",
        "",
        "## Source ledger",
        "",
        "| Source class | Family | Product | Cadence / coverage | Disposition | Exact reason |",
        "|---|---|---|---|---|---|",
    ]
    for row in ledger.to_dict("records"):
        coverage = f"{row['source_interval']}; {row['maximum_observed_staleness']}"
        lines.append(f"| `{row['source_id']}` | {row['family']} | {row['product_type']} | {coverage} | `{row['disposition']}` | {row['evidence']} |")
    lines.extend([
        "",
        "## Causal blockers",
        "",
        "Historical event timestamps do not prove observation or availability timestamps. Mixed sidecars discard provider, product, revision, and capture identity. Funding settlement-versus-estimate semantics are absent. Historical OI mixes native/base and quote-notional lineage. Replay and embedded OHLCV paths apply forward fill without a maximum age.",
        "",
        "The forbidden A7 inputs are: " + ", ".join(f"`{item}`" for item in FORBIDDEN_FUTURE_FUNDING) + ".",
        "",
        "## Admission repair required",
        "",
        "- Persist immutable provider, exchange, market ID, PF/PI product kind, event timestamp, observation timestamp, availability timestamp, ingestion timestamp, cadence, publication/settlement meaning, revision/version, units, and raw-payload hash per observation.",
        "- Keep export, API, analytics, and live captures in source-specific tables; never merge away provenance.",
        "- Establish availability and publication delay from a documented source contract or timestamped capture, never convenience inference.",
        "- Use only PF linear-USD products for the frozen candidate contract and prove native/base-to-quote OI conversion units.",
        "- Use `available_ts <= action_decision_ts` as-of joins, predeclare a finite maximum age by source, and reject missing/stale rows rather than forward filling without limit.",
        "- Demonstrate historical/live field, product, cadence, timestamp, and unit parity. For A7, prove that the value is the last known published/settled rate and never a future estimate, next payment, future settlement, or revised future value.",
        "- Re-run month/symbol/side coverage and staleness tests on the exact Stage-D action population before admitting A6 or A7.",
        "",
        "## Executable availability checks",
        "",
        f"All checks passed: `{str(tests['passed']).lower()}`. Source classes: {tests['source_classes']}; admitted: {tests['admitted_sources']}.",
        "",
    ])
    return "\n".join(lines)


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"fresh output root required: {output}")
    output.mkdir(parents=True)
    ledger = build_ledger()
    tests = validate_ledger(ledger)
    ledger_path = output / "oi_funding_source_ledger.parquet"
    tests_path = output / "oi_funding_availability_tests.json"
    report_path = output / "oi_funding_lineage_report.md"
    _atomic_write_parquet(ledger_path, ledger)
    _atomic_write_text(tests_path, json.dumps(tests, indent=2, sort_keys=True) + "\n")
    _atomic_write_text(report_path, _report(ledger, tests))

    legacy_source_files = [
        path
        for directory in LEGACY_SOURCE_DIRS
        for path in sorted(directory.glob("*.parquet"))
    ]
    source_files = [FUNDING_ZIP, REFERENCE_ZIP, *sorted(FUNDING_DIR.glob("*.parquet")), *sorted(OI_DIR.glob("*.parquet")), *legacy_source_files, *CODE_EVIDENCE]
    inputs = {str(path.relative_to(ROOT)): sha256(path) for path in source_files}
    script_path = Path(__file__).resolve()
    outputs = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    manifest = {
        "schema": "stage_d_oi_funding_lineage_manifest_v4",
        "status": "SEALED_REJECTED_LINEAGE",
        "a6_disposition": "REJECTED_LINEAGE",
        "a7_disposition": "REJECTED_LINEAGE",
        "companion_disposition": "OI_FUNDING_CAUSAL_LINEAGE_UNRESOLVED",
        "allowed_source_dispositions": list(ALLOWED_DISPOSITIONS),
        "read_only_source_audit": True,
        "feature_admission_changed": False,
        "inputs_sha256": inputs,
        "code_sha256": {str(script_path.relative_to(ROOT)): sha256(script_path)},
        "outputs_sha256": outputs,
        "manifest_self_hash_excluded": True,
    }
    _atomic_write_text(output / "run_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    persisted = json.loads((output / "run_manifest.json").read_text())
    for name, expected in persisted["outputs_sha256"].items():
        if sha256(output / name) != expected:
            raise ValueError(f"sealed output mismatch: {name}")
    return {"output": str(output), "source_classes": len(ledger), "inputs_hashed": len(inputs), "outputs_hashed": len(outputs), "tests": tests}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
