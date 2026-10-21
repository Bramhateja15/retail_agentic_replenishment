"""
Phase E diagnostic (read-only) — row-level impact of the V2Locations
field-count anomaly: how many GKG records contain at least one malformed
block, as a share of all records.

Does NOT modify raw data. Output is saved as JSON and is read by
07_clean_gdelt.py so the cleaning report uses measured values.

Output:
    results/inspection/gdelt_v2locations_row_impact_report.json

Run (from any folder):
    python scripts/06c_diagnose_v2locations_row_impact.py
"""

import csv
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import RAW_GDELT_DIR, INSPECTION_DIR, ensure_dirs

V2LOCATIONS_INDEX = 10
EXPECTED_FIELD_COUNT = 27


def set_csv_field_limit():
    max_field_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_field_size)
            break
        except OverflowError:
            max_field_size //= 10


def main():
    ensure_dirs()
    set_csv_field_limit()

    all_files = sorted(RAW_GDELT_DIR.rglob("*.gkg.csv.zip"))
    if not all_files:
        print(f"[STOP] No raw GDELT files found under {RAW_GDELT_DIR}.")
        sys.exit(1)

    total_rows = 0
    affected_rows = 0
    total_malformed_blocks = 0

    for fp in all_files:
        with zipfile.ZipFile(fp) as z:
            with z.open(z.namelist()[0]) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                for row in csv.reader(text, delimiter="\t"):
                    if len(row) != EXPECTED_FIELD_COUNT:
                        continue
                    total_rows += 1
                    loc_val = row[V2LOCATIONS_INDEX]
                    if loc_val == "":
                        continue
                    row_has_malformed_block = False
                    for block in loc_val.split(";"):
                        if block == "":
                            continue
                        if len(block.split("#")) != 9:
                            total_malformed_blocks += 1
                            row_has_malformed_block = True
                    if row_has_malformed_block:
                        affected_rows += 1

    pct = (affected_rows / total_rows) * 100 if total_rows else 0.0

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_examined": len(all_files),
        "rows_examined": total_rows,
        "total_malformed_blocks": total_malformed_blocks,
        "affected_rows": affected_rows,
        "affected_row_percentage": round(pct, 4),
    }
    out_path = INSPECTION_DIR / "gdelt_v2locations_row_impact_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Total rows examined:              {total_rows}")
    print(f"Total malformed V2Locations blocks: {total_malformed_blocks}")
    print(f"Distinct rows containing >=1 malformed block: {affected_rows}")
    print(f"Affected-row percentage:           {pct:.4f}%")
    print(f"\nReport written to:\n  {out_path}")


if __name__ == "__main__":
    main()
