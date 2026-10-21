"""
Phase E diagnostic (read-only) — decompose the V2Locations rows flagged
by 06_validate_gdelt_corpus.py by failure mode:
  (a) wrong field count (block does not have 9 '#'-delimited fields), vs
  (b) correct field count but undocumented Location Type (not in 1-5).

Does NOT modify raw data. Results are compared with the full-corpus
validation report (if present) and saved as JSON so the finding is auditable.

Output:
    results/inspection/gdelt_v2locations_failure_modes_report.json

Run (from any folder):
    python scripts/06b_diagnose_v2locations.py
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

VALID_TYPES = {"1", "2", "3", "4", "5"}
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


def expected_from_full_validation():
    """Affected-row count found by 06_validate_gdelt_corpus.py, if available."""
    path = INSPECTION_DIR / "gdelt_full_corpus_validation.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["checks"]["v2locations_failures"]["count"]


def main():
    ensure_dirs()
    set_csv_field_limit()

    all_files = sorted(RAW_GDELT_DIR.rglob("*.gkg.csv.zip"))
    if not all_files:
        print(f"[STOP] No raw GDELT files found under {RAW_GDELT_DIR}.")
        sys.exit(1)

    total_rows = 0
    rows_field_count_only = 0
    rows_type_only = 0
    rows_both = 0
    field_count_blocks = 0
    type_only_blocks = 0

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

                    has_field_count_issue = False
                    has_type_issue = False
                    for block in loc_val.split(";"):
                        if block == "":
                            continue
                        parts = block.split("#")
                        if len(parts) != 9:
                            field_count_blocks += 1
                            has_field_count_issue = True
                        elif parts[0] not in VALID_TYPES:
                            type_only_blocks += 1
                            has_type_issue = True

                    if has_field_count_issue and has_type_issue:
                        rows_both += 1
                    elif has_field_count_issue:
                        rows_field_count_only += 1
                    elif has_type_issue:
                        rows_type_only += 1

    total_affected = rows_field_count_only + rows_type_only + rows_both
    expected = expected_from_full_validation()

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_examined": len(all_files),
        "rows_examined": total_rows,
        "rows_failing_field_count_only": rows_field_count_only,
        "rows_failing_location_type_only": rows_type_only,
        "rows_failing_both": rows_both,
        "total_affected_rows": total_affected,
        "malformed_blocks_field_count": field_count_blocks,
        "malformed_blocks_type_only": type_only_blocks,
        "total_malformed_blocks": field_count_blocks + type_only_blocks,
        "matches_full_corpus_validation": (None if expected is None else total_affected == expected),
        "full_corpus_validation_affected_rows": expected,
    }
    out_path = INSPECTION_DIR / "gdelt_v2locations_failure_modes_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Total rows examined: {total_rows}")
    print(f"Rows failing ONLY on field-count (!=9): {rows_field_count_only}")
    print(f"Rows failing ONLY on Location Type (9 fields, type not in 1-5): {rows_type_only}")
    print(f"Rows failing on BOTH modes: {rows_both}")
    print(f"Total distinct affected rows: {total_affected}"
          + (f"  (full-corpus validation found: {expected})" if expected is not None else ""))
    print(f"\nMalformed blocks by mode — field-count: {field_count_blocks}, type-only: {type_only_blocks}")
    print(f"Total malformed blocks: {field_count_blocks + type_only_blocks}")
    print(f"\nReport written to:\n  {out_path}")


if __name__ == "__main__":
    main()
