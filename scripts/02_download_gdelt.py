"""
Phase B — Raw data acquisition: GDELT 2.0 GKG

Downloads raw GDELT 2.0 Global Knowledge Graph (GKG) files covering the
FreshRetailNet date range (plus a configurable buffer), at a sampled
number of 15-minute snapshots per day (default: 4/day) to keep raw
volume proportionate.

This script does NOT parse GKG field contents (themes/locations/tone
extraction happens in Phase E cleaning). It only fetches and stores the
raw .csv.zip files exactly as published by GDELT.

Date range resolution order:
    1. --start / --end command-line arguments (highest priority)
    2. gdelt.start_date / gdelt.end_date in config/project.yaml, if set
    3. Auto-detection from FreshRetailNet raw parquet column statistics
       (read-only; does not modify FreshRetailNet raw data)

If none of these resolve unambiguously, the script stops and reports
what it found, rather than guessing.

Run:
    python scripts/02_download_gdelt.py --dry-run
    python scripts/02_download_gdelt.py
    python scripts/02_download_gdelt.py --start 2024-01-01 --end 2024-03-31
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import yaml
import pyarrow.parquet as pq

from config.paths import RAW_FRESHRETAILNET_DIR, RAW_GDELT_DIR, METADATA_DIR, ensure_dirs

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "project.yaml"
GDELT_BASE_URL = "http://data.gdeltproject.org/gdeltv2"

DATE_COLUMN_CANDIDATES = [
    "dt", "date", "stat_date", "sale_date", "ds", "day",
    "event_date", "order_date", "timestamp", "time",
]


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------
# Date-range resolution
# --------------------------------------------------------------------

def find_parquet_files():
    files = sorted(RAW_FRESHRETAILNET_DIR.rglob("*.parquet"))
    return files


def detect_date_column(files):
    if not files:
        return []
    schema = pq.ParquetFile(files[0]).schema_arrow
    names_lower = {n.lower(): n for n in schema.names}
    return [names_lower[c] for c in DATE_COLUMN_CANDIDATES if c in names_lower]


def try_stats_min_max(files, column):
    """Attempt to read min/max from parquet row-group statistics only
    (fast, no full column scan). Returns (None, None) if unavailable."""
    gmin, gmax = None, None
    try:
        for fp in files:
            pf = pq.ParquetFile(fp)
            names = pf.schema_arrow.names
            if column not in names:
                continue
            col_idx = names.index(column)
            for rg in range(pf.num_row_groups):
                stats = pf.metadata.row_group(rg).column(col_idx).statistics
                if stats is None or not stats.has_min_max:
                    return None, None
                mn, mx = stats.min, stats.max
                gmin = mn if gmin is None or mn < gmin else gmin
                gmax = mx if gmax is None or mx > gmax else gmax
        return gmin, gmax
    except Exception:
        return None, None


def try_pandas_min_max(files, column):
    import pandas as pd
    mins, maxs = [], []
    for fp in files:
        try:
            s = pd.read_parquet(fp, columns=[column])[column]
        except Exception:
            continue
        s = pd.to_datetime(s, errors="coerce")
        if s.notna().any():
            mins.append(s.min())
            maxs.append(s.max())
    if not mins:
        return None, None
    return min(mins), max(maxs)


def auto_detect_date_range():
    files = find_parquet_files()
    if not files:
        print("[AUTO-DETECT] No FreshRetailNet parquet files found under "
              f"{RAW_FRESHRETAILNET_DIR}. Run 01_download_freshretailnet.py first.")
        return None, None, "no_raw_files"

    candidates = detect_date_column(files)
    if len(candidates) == 0:
        print("[AUTO-DETECT] No obvious date column found. Available columns:")
        print("  ", pq.ParquetFile(files[0]).schema_arrow.names)
        return None, None, "no_candidate_column"
    if len(candidates) > 1:
        print(f"[AUTO-DETECT] Multiple possible date columns found: {candidates}. "
              "Ambiguous — please specify manually.")
        return None, None, "ambiguous_columns"

    column = candidates[0]
    print(f"[AUTO-DETECT] Using column '{column}' as the date field.")

    mn, mx = try_stats_min_max(files, column)
    if mn is None:
        print("[AUTO-DETECT] Row-group statistics unavailable — falling back to full column read.")
        mn, mx = try_pandas_min_max(files, column)

    if mn is None:
        print(f"[AUTO-DETECT] Could not derive a valid min/max from column '{column}'.")
        return None, None, "unparseable_values"

    return mn, mx, f"auto_detected_from_column:{column}"


def resolve_date_range(args, cfg):
    if args.start and args.end:
        return (datetime.strptime(args.start, "%Y-%m-%d").date(),
                datetime.strptime(args.end, "%Y-%m-%d").date(),
                "cli_arguments")

    gdelt_cfg = cfg.get("gdelt", {})
    if gdelt_cfg.get("start_date") and gdelt_cfg.get("end_date"):
        return (datetime.strptime(str(gdelt_cfg["start_date"]), "%Y-%m-%d").date(),
                datetime.strptime(str(gdelt_cfg["end_date"]), "%Y-%m-%d").date(),
                "config_project_yaml")

    mn, mx, method = auto_detect_date_range()
    if mn is None:
        return None, None, method

    def to_date(v):
        if hasattr(v, "date"):
            return v.date()
        return datetime.strptime(str(v)[:10], "%Y-%m-%d").date()

    return to_date(mn), to_date(mx), method


# --------------------------------------------------------------------
# GDELT download
# --------------------------------------------------------------------

def build_targets(start_date, end_date, buffer_days, snapshot_times):
    targets = []
    d = start_date - timedelta(days=buffer_days)
    end = end_date + timedelta(days=buffer_days)
    while d <= end:
        for hhmmss in snapshot_times:
            stamp = f"{d.strftime('%Y%m%d')}{hhmmss}"
            targets.append((d, stamp))
        d += timedelta(days=1)
    return targets


def download_file(session, stamp, out_path, retries=3, timeout=30):
    url = f"{GDELT_BASE_URL}/{stamp}.gkg.csv.zip"
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200:
                out_path.write_bytes(resp.content)
                return True, len(resp.content), None
            elif resp.status_code == 404:
                return False, 0, "not_found_on_server"
            else:
                last_err = f"http_{resp.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(1.5 * attempt)
    return False, 0, last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap number of files fetched (testing only)")
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_config()
    gdelt_cfg = cfg.get("gdelt", {})
    buffer_days = int(gdelt_cfg.get("buffer_days", 3))
    snapshot_times = gdelt_cfg.get("snapshot_times", ["000000", "060000", "120000", "180000"])

    start_date, end_date, method = resolve_date_range(args, cfg)

    if start_date is None:
        print("\n[STOP] Could not resolve a date range. Options:")
        print("  1) Run: python scripts/02_download_gdelt.py --start YYYY-MM-DD --end YYYY-MM-DD")
        print("  2) Or set gdelt.start_date / gdelt.end_date in config/project.yaml")
        print(f"  (Reason: {method})")
        sys.exit(1)

    print(f"Resolved date range: {start_date} to {end_date}  (method: {method})")
    print(f"Buffer: {buffer_days} days each side | Snapshots/day: {len(snapshot_times)} {snapshot_times}")

    targets = build_targets(start_date, end_date, buffer_days, snapshot_times)
    if args.limit:
        targets = targets[: args.limit]

    print(f"Planned raw GKG files to fetch: {len(targets)}")

    if args.dry_run:
        print("[DRY RUN] No files downloaded. Example URLs:")
        for d, stamp in targets[:5]:
            print(f"  {GDELT_BASE_URL}/{stamp}.gkg.csv.zip")
        return

    session = requests.Session()
    succeeded, skipped, failed = [], [], []

    for d, stamp in targets:
        day_dir = RAW_GDELT_DIR / d.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        out_path = day_dir / f"{stamp}.gkg.csv.zip"

        if out_path.exists() and out_path.stat().st_size > 0:
            skipped.append(stamp)
            continue

        ok, nbytes, err = download_file(session, stamp, out_path)
        if ok:
            succeeded.append({"stamp": stamp, "bytes": nbytes})
        else:
            failed.append({"stamp": stamp, "reason": err})
        time.sleep(0.2)  # be polite to the server

    manifest = {
        "source": "GDELT 2.0 GKG",
        "base_url": GDELT_BASE_URL,
        "date_range_start": str(start_date),
        "date_range_end": str(end_date),
        "date_range_resolution_method": method,
        "buffer_days": buffer_days,
        "snapshot_times": snapshot_times,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "planned_count": len(targets),
        "succeeded_count": len(succeeded),
        "skipped_already_present_count": len(skipped),
        "failed_count": len(failed),
        "failed_details": failed,
    }

    manifest_path = METADATA_DIR / "gdelt_raw_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSucceeded: {len(succeeded)} | Skipped (already present): {len(skipped)} | Failed: {len(failed)}")
    print(f"Raw files stored under: {RAW_GDELT_DIR}")
    print(f"Manifest written to: {manifest_path}")
    if failed:
        print("Note: some failures are expected — GDELT occasionally has gaps in its 15-minute publication schedule.")


if __name__ == "__main__":
    main()