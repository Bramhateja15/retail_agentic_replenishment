"""
Phase E diagnostic (read-only, ad hoc) — decompose the 307 affected
V2Locations rows by failure mode: (a) wrong field count, vs (b) correct
field count but undocumented Location Type. Does NOT modify raw data.
"""

import zipfile
import io
import csv
from pathlib import Path
import sys 

max_field_size = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_field_size)
        break
    except OverflowError:
        max_field_size //= 10

RAW_GDELT_DIR = Path("data/raw/gdelt")
all_files = sorted(RAW_GDELT_DIR.rglob("*.gkg.csv.zip"))
VALID_TYPES = {"1", "2", "3", "4", "5"}

rows_with_field_count_failure = 0
rows_with_type_failure_only = 0   # 9 fields present, but type not in {1..5}
rows_with_both_failure_modes = 0
total_rows = 0

field_count_blocks = 0
type_only_blocks = 0

for fp in all_files:
    with zipfile.ZipFile(fp) as z:
        with z.open(z.namelist()[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            reader = csv.reader(text, delimiter="\t")
            for row in reader:
                if len(row) != 27:
                    continue
                total_rows += 1
                loc_val = row[10]
                if loc_val == "":
                    continue

                row_has_field_count_issue = False
                row_has_type_only_issue = False

                for block in loc_val.split(";"):
                    if block == "":
                        continue
                    parts = block.split("#")
                    if len(parts) != 9:
                        field_count_blocks += 1
                        row_has_field_count_issue = True
                    elif parts[0] not in VALID_TYPES:
                        type_only_blocks += 1
                        row_has_type_only_issue = True

                if row_has_field_count_issue and row_has_type_only_issue:
                    rows_with_both_failure_modes += 1
                elif row_has_field_count_issue:
                    rows_with_field_count_failure += 1
                elif row_has_type_only_issue:
                    rows_with_type_failure_only += 1

print(f"Total rows examined: {total_rows}")
print(f"Rows failing ONLY on field-count (!=9): {rows_with_field_count_failure}")
print(f"Rows failing ONLY on Location Type (9 fields, type not in 1-5): {rows_with_type_failure_only}")
print(f"Rows failing on BOTH modes: {rows_with_both_failure_modes}")
total_affected = rows_with_field_count_failure + rows_with_type_failure_only + rows_with_both_failure_modes
print(f"Total distinct affected rows: {total_affected}  (expect: 307)")
print(f"\nMalformed blocks by mode — field-count: {field_count_blocks}, type-only: {type_only_blocks}")
print(f"Total malformed blocks: {field_count_blocks + type_only_blocks}  (expect: 2850)")