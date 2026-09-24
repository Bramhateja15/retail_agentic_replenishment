import zipfile
import io
import csv
import sys
from pathlib import Path

max_field_size = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_field_size)
        break
    except OverflowError:
        max_field_size //= 10

RAW_GDELT_DIR = Path("data/raw/gdelt")
all_files = sorted(RAW_GDELT_DIR.rglob("*.gkg.csv.zip"))

total_rows = 0
affected_rows = 0
total_malformed_blocks = 0

for fp in all_files:
    with zipfile.ZipFile(fp) as z:
        with z.open(z.namelist()[0]) as raw:
            text = io.TextIOWrapper(
                raw,
                encoding="utf-8",
                errors="replace"
            )
            reader = csv.reader(text, delimiter="\t")

            for row in reader:
                if len(row) != 27:
                    continue

                total_rows += 1
                loc_val = row[10]

                if loc_val == "":
                    continue

                row_has_malformed_block = False

                for block in loc_val.split(";"):
                    if block == "":
                        continue

                    parts = block.split("#")

                    if len(parts) != 9:
                        total_malformed_blocks += 1
                        row_has_malformed_block = True

                if row_has_malformed_block:
                    affected_rows += 1

pct = (affected_rows / total_rows) * 100 if total_rows else 0.0

print(f"Total rows examined:              {total_rows}")
print(f"Total malformed V2Locations blocks: {total_malformed_blocks}")
print(f"Distinct rows containing >=1 malformed block: {affected_rows}")
print(f"Affected-row percentage:           {pct:.4f}%")