#!/usr/bin/env python3
"""Report daily old-versus-new meta top-k outcomes on one aligned universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["__ts__", "__symbol__", "side_name"]
LABEL_COLUMNS = [
    *KEYS,
    "__first_touch_capture_net__",
    "__first_touch_valid_path__",
    "__first_touch_mae_norm__",
    "__first_touch_full_path_mae_norm__",
    "__first_touch_timeout__",
    "__mfe_1r_before_mae_1r__",
]


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _read_labels(labels_dir: Path, priority: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for side in ("long", "short"):
        path = labels_dir / f"train_global_{side}_5_2026_07.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=LABEL_COLUMNS)
        frame["__ts__"] = _utc(frame["__ts__"])
        frame["_label_priority"] = np.int8(priority)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No July long/short label files found under {labels_dir}"
        )
    return pd.concat(frames, ignore_index=True, copy=False)


def _outcomes_from_labels(old_dir: Path, new_dir: Path) -> pd.DataFrame:
    # Preserve validated historical outcomes and use the refreshed artifact only
    # for timestamps that were previously unavailable.
    labels = pd.concat(
        [_read_labels(old_dir, 0), _read_labels(new_dir, 1)],
        ignore_index=True,
        copy=False,
    )
    labels = labels.sort_values("_label_priority").drop_duplicates(KEYS, keep="first")
    valid = pd.to_numeric(labels["__first_touch_valid_path__"], errors="coerce").gt(0.5)
    capture_net = pd.to_numeric(labels["__first_touch_capture_net__"], errors="coerce")
    first_mae = pd.to_numeric(labels["__first_touch_mae_norm__"], errors="coerce")
    full_mae = pd.to_numeric(
        labels["__first_touch_full_path_mae_norm__"], errors="coerce"
    )
    timeout = pd.to_numeric(labels["__first_touch_timeout__"], errors="coerce").fillna(
        0.0
    )
    mfe_before_mae = pd.to_numeric(
        labels["__mfe_1r_before_mae_1r__"], errors="coerce"
    ).fillna(0.0)

    # Match the established S52/S59 reporting contract exactly: the trailing
    # capture already embeds 1% round-trip cost and the executable floor applies
    # the same additional comparison convention used by the frozen report.
    exec_margin = capture_net
    ev_after_1pct = capture_net - 0.01
    clean_exec = (
        exec_margin.gt(0.0)
        & first_mae.lt(1.0)
        & timeout.lt(0.5)
        & mfe_before_mae.gt(0.5)
    )
    out = labels.loc[:, KEYS].copy()
    out["ev_after_1pct"] = ev_after_1pct.where(valid)
    out["clean_exec"] = clean_exec.astype(np.float32).where(valid)
    out["first_touch_bad_mae_1r"] = first_mae.ge(1.0).astype(np.float32).where(valid)
    out["full_path_bad_mae_1r"] = full_mae.ge(1.0).astype(np.float32).where(valid)
    out["timeout"] = timeout.gt(0.5).astype(np.float32).where(valid)
    out["label_source"] = np.where(
        labels["_label_priority"].eq(0), "validated_old", "refreshed_tail"
    )
    return out


def _standardize_historical(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["__ts__"] = _utc(frame["__ts__"])
    frame = frame.rename(columns={"score_current_reference": "score_old"})
    for candidate in ("score_adjusted", "score_shock_adjusted", "score_alternative"):
        if candidate in frame.columns:
            frame = frame.rename(columns={candidate: "score_new"})
            break
    if "score_new" not in frame.columns:
        raise KeyError(
            "Historical predictions contain no recognized adjusted-score column"
        )
    return frame


def _standardize_july_aligned(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["__ts__"] = _utc(frame["__ts__"])
    return frame.rename(
        columns={
            "score_meta_base_soft_label": "score_old",
            "score_shock_adjusted": "score_new",
        }
    )


def _standardize_july_extension(path: Path, outcomes: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["__ts__"] = _utc(frame["__ts__"])
    if "decision_ts" in frame.columns:
        frame["decision_ts"] = _utc(frame["decision_ts"])
        frame = frame.sort_values("decision_ts").drop_duplicates(KEYS, keep="last")
    else:
        frame = frame.drop_duplicates(KEYS, keep="last")
    frame = frame.rename(
        columns={
            "score_current_reference": "score_old",
            "score_shock_adjusted": "score_new",
        }
    )
    stale = [
        col
        for col in (
            "ev_after_1pct",
            "clean_exec",
            "first_touch_bad_mae_1r",
            "full_path_bad_mae_1r",
            "timeout",
            "label_source",
        )
        if col in frame.columns
    ]
    existing = frame[KEYS + stale].copy() if stale else frame[KEYS].copy()
    merged = frame.drop(columns=stale).merge(
        outcomes, on=KEYS, how="left", validate="one_to_one"
    )
    if stale:
        existing = existing.rename(
            columns={name: f"{name}__extension" for name in stale}
        )
        merged = merged.merge(existing, on=KEYS, how="left", validate="one_to_one")
        for name in stale:
            extension_name = f"{name}__extension"
            if name in merged.columns:
                merged[name] = merged[extension_name].combine_first(merged[name])
            else:
                merged[name] = merged[extension_name]
            merged = merged.drop(columns=[extension_name])
    return merged


def _rank_within_timestamp(frame: pd.DataFrame, score: str) -> pd.Series:
    numeric = pd.to_numeric(frame[score], errors="coerce")
    return numeric.groupby(frame["__ts__"], sort=False).rank(method="average", pct=True)


def _daily_metrics(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    scoped = frame.loc[frame["__ts__"].between(start, end, inclusive="both")].copy()
    scoped = scoped.drop_duplicates(KEYS, keep="last")
    scoped["day"] = scoped["__ts__"].dt.strftime("%Y-%m-%d")
    scoped["rank_old"] = _rank_within_timestamp(scoped, "score_old")
    scoped["rank_new"] = _rank_within_timestamp(scoped, "score_new")

    days = pd.date_range(start.floor("D"), end.floor("D"), freq="D", tz="UTC")
    records: list[dict[str, float | int | str]] = []
    for day in days:
        day_name = day.strftime("%Y-%m-%d")
        daily = scoped.loc[scoped["day"].eq(day_name)]
        record: dict[str, float | int | str] = {
            "date": day_name,
            "prediction_rows": int(len(daily)),
            "hours_covered": int(daily["__ts__"].dt.floor("h").nunique()),
            "outcome_rows": int(
                pd.to_numeric(daily["ev_after_1pct"], errors="coerce").notna().sum()
            ),
        }
        for arm in ("old", "new"):
            selected = daily.loc[daily[f"rank_{arm}"].ge(0.90)].copy()
            selected = selected.loc[
                pd.to_numeric(selected["ev_after_1pct"], errors="coerce").notna()
            ]
            record[f"{arm}_trades"] = int(len(selected))
            record[f"{arm}_ev"] = float(
                pd.to_numeric(selected["ev_after_1pct"], errors="coerce").mean()
            )
            record[f"{arm}_hit_rate"] = float(
                pd.to_numeric(selected["clean_exec"], errors="coerce").mean()
            )
        record["delta_ev"] = float(record["new_ev"]) - float(record["old_ev"])
        record["delta_hit_rate"] = float(record["new_hit_rate"]) - float(
            record["old_hit_rate"]
        )
        records.append(record)
    return pd.DataFrame.from_records(records), scoped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-predictions", type=Path, required=True)
    parser.add_argument("--july-aligned-predictions", type=Path, required=True)
    parser.add_argument(
        "--july-extension-predictions",
        type=Path,
        nargs="+",
        required=True,
        help="One or more extensions in ascending provenance priority; later files win.",
    )
    parser.add_argument("--old-july-labels-dir", type=Path, required=True)
    parser.add_argument("--new-july-labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", default="2026-06-28T00:00:00Z")
    parser.add_argument("--end", default="2026-07-10T23:59:59Z")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    outcomes = _outcomes_from_labels(args.old_july_labels_dir, args.new_july_labels_dir)
    frames = [_standardize_historical(args.historical_predictions)]
    frames.extend(
        _standardize_july_extension(path, outcomes)
        for path in args.july_extension_predictions
    )
    # The original materialized OOS research predictions are authoritative on
    # overlap; reconstructed extensions only fill absent batches.
    frames.append(_standardize_july_aligned(args.july_aligned_predictions))
    required = [*KEYS, "score_old", "score_new", "ev_after_1pct", "clean_exec"]
    combined = pd.concat(
        [frame.loc[:, required] for frame in frames], ignore_index=True, copy=False
    )
    daily, aligned = _daily_metrics(combined, start, end)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = args.output_dir / "daily_old_new_global_top10.csv"
    aligned_path = args.output_dir / "aligned_old_new_rows.parquet"
    manifest_path = args.output_dir / "manifest.json"
    daily.to_csv(daily_path, index=False)
    aligned.to_parquet(aligned_path, index=False)
    manifest = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "selection_contract": "global within-timestamp top 10% of the fixed base-to-meta candidate universe",
        "ev_contract": "S52/S59 ev_after_1pct reporting convention",
        "hit_rate_contract": "clean_exec precision",
        "daily_metrics": str(daily_path),
        "aligned_rows": str(aligned_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(daily.to_string(index=False))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
