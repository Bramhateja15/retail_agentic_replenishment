"""
Phase C — Raw data inspection: GDELT 2.0 GKG

Read-only inspection of the raw .csv.zip files. Two passes:

  1. Corpus-wide integrity pass (ALL files): confirms each zip opens,
     contains exactly one member, and is non-empty. Cheap — does not
     decompress content.

  2. Detailed structural/content pass (SAMPLED files: the 00:00 UTC
     snapshot for each day, i.e. 1 of the 4 daily files — a systematic,
     documented sample, not the full corpus, to keep runtime bounded).
     Parses each sampled file against the published GDELT 2.1 GKG
     27-column layout, reports actual field-count distribution
     (evidence of how "semi-structured" the raw source really is),
     date-field validity, and prevalence of non-empty theme/location/
     person/organisation/tone fields.

Nothing is cleaned, filtered, or written back to data/raw/.

Important finding:-
The structural pass found the outer 27-field TSV structure to be
fully well-formed across the sample, with 0 malformed rows and
0 invalid DATE fields. Therefore, the preparation requirement for
GDELT is not based on row-level corruption. Instead, the content
fields such as V2Themes, V2Locations, V2Persons and V2Organizations
contain compound, multi-valued encodings that require deliberate
parsing, extraction and aggregation before they can be used as
structured analytical features.

Run:
    python scripts/04_inspect_gdelt.py
"""

import csv
import io
import json
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import RAW_GDELT_DIR, INSPECTION_DIR, ensure_dirs

# Published GDELT 2.1 GKG column layout (per the official GDELT 2.1 GKG
# codebook). This is a REFERENCE for the inspection pass only — it is
# not yet used to reshape or clean anything. Phase E confirms and acts
# on whichever of this actually matches the observed data.
GKG_V2_1_COLUMNS = [
    "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName",
    "DocumentIdentifier", "Counts", "V2Counts", "Themes", "V2Themes",
    "Locations", "V2Locations", "Persons", "V2Persons", "Organizations",
    "V2Organizations", "V2Tone", "Dates", "GCAM", "SharingImage",
    "RelatedImages", "SocialImageEmbeds", "SocialVideoEmbeds", "Quotations",
    "AllNames", "Amounts", "TranslationInfo", "Extras",
]
EXPECTED_FIELD_COUNT = len(GKG_V2_1_COLUMNS)  # 27

CONTENT_FIELDS_TO_CHECK = ["V2Themes", "V2Locations", "V2Persons",
                           "V2Organizations", "V2Tone"]


def integrity_pass(all_files):
    results = []
    for fp in all_files:
        entry = {"file": str(fp.relative_to(RAW_GDELT_DIR)), "size_bytes": fp.stat().st_size}
        try:
            with zipfile.ZipFile(fp) as z:
                names = z.namelist()
                entry["member_count"] = len(names)
                entry["opens_ok"] = True
                entry["uncompressed_bytes"] = sum(i.file_size for i in z.infolist())
        except zipfile.BadZipFile:
            entry["opens_ok"] = False
            entry["member_count"] = 0
            entry["uncompressed_bytes"] = 0
        results.append(entry)
    return results


def read_zip_as_rows(fp):
    with zipfile.ZipFile(fp) as z:
        names = z.namelist()
        if not names:
            return []
        with z.open(names[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            reader = csv.reader(text, delimiter="\t")
            return list(reader)


def structural_pass(sample_files):
    field_count_distribution = Counter()
    n_rows_total = 0
    n_files_parsed = 0
    n_files_unreadable = 0
    date_valid = 0
    date_invalid = 0
    content_nonempty_counts = Counter()
    per_file_row_counts = []

    for fp in sample_files:
        try:
            rows = read_zip_as_rows(fp)
        except Exception:
            n_files_unreadable += 1
            continue

        n_files_parsed += 1
        per_file_row_counts.append({"file": str(fp.relative_to(RAW_GDELT_DIR)), "rows": len(rows)})
        n_rows_total += len(rows)

        for row in rows:
            field_count_distribution[len(row)] += 1

            # DATE field (index 1) expected as 14-digit YYYYMMDDHHMMSS
            if len(row) > 1:
                date_field = row[1]
                if len(date_field) == 14 and date_field.isdigit():
                    date_valid += 1
                else:
                    date_invalid += 1

            # Content field prevalence, only checked where the row
            # actually has enough fields to index safely
            if len(row) == EXPECTED_FIELD_COUNT:
                for fname in CONTENT_FIELDS_TO_CHECK:
                    idx = GKG_V2_1_COLUMNS.index(fname)
                    if row[idx].strip():
                        content_nonempty_counts[fname] += 1

    return {
        "files_sampled": len(sample_files),
        "files_parsed_ok": n_files_parsed,
        "files_unreadable": n_files_unreadable,
        "total_rows_in_sample": n_rows_total,
        "field_count_distribution": dict(field_count_distribution),
        "expected_field_count": EXPECTED_FIELD_COUNT,
        "rows_matching_expected_field_count": field_count_distribution.get(EXPECTED_FIELD_COUNT, 0),
        "date_field_valid_rows": date_valid,
        "date_field_invalid_rows": date_invalid,
        "content_field_nonempty_row_counts": dict(content_nonempty_counts),
        "per_file_row_counts_sample": per_file_row_counts[:10],  # preview only
    }


def main():
    ensure_dirs()

    all_files = sorted(RAW_GDELT_DIR.rglob("*.gkg.csv.zip"))
    if not all_files:
        print(f"[STOP] No raw GDELT files found under {RAW_GDELT_DIR}.")
        print("Run scripts/02_download_gdelt.py first.")
        sys.exit(1)

    print(f"Found {len(all_files)} raw GDELT files. Running integrity pass (all files)...")
    integrity_results = integrity_pass(all_files)
    n_bad = sum(1 for r in integrity_results if not r["opens_ok"])
    n_empty = sum(1 for r in integrity_results if r["opens_ok"] and r["uncompressed_bytes"] == 0)

    # Sampling strategy: the 00:00 UTC snapshot from each day folder —
    # systematic (not random), documented, ~1/4 of the corpus.
    day_dirs = sorted(d for d in RAW_GDELT_DIR.iterdir() if d.is_dir())
    sample_files = []
    for d in day_dirs:
        candidates = sorted(d.glob("*000000.gkg.csv.zip"))
        if candidates:
            sample_files.append(candidates[0])

    print(f"Running structural/content pass on {len(sample_files)} sampled files "
          f"(00:00 UTC snapshot per day)...")
    structural_results = structural_pass(sample_files)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gkg_reference_schema": GKG_V2_1_COLUMNS,
        "integrity_pass": {
            "total_files": len(all_files),
            "files_failing_to_open": n_bad,
            "files_zero_bytes_uncompressed": n_empty,
        },
        "structural_pass": structural_results,
        "sampling_method": "one file per day (00:00 UTC snapshot), "
                            f"{len(sample_files)} of {len(all_files)} total raw files",
    }

    out_path = INSPECTION_DIR / "gdelt_inspection_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== GDELT INSPECTION SUMMARY ===")
    print(f"Total raw files: {len(all_files)}")
    print(f"Files failing to open: {n_bad}")
    print(f"Files with zero uncompressed bytes: {n_empty}")
    print(f"Sampled files parsed OK: {structural_results['files_parsed_ok']} / {structural_results['files_sampled']}")
    print(f"Total rows in sample: {structural_results['total_rows_in_sample']}")
    print(f"Field-count distribution (expect mostly {EXPECTED_FIELD_COUNT}): "
          f"{structural_results['field_count_distribution']}")
    print(f"Rows matching expected {EXPECTED_FIELD_COUNT}-field layout: "
          f"{structural_results['rows_matching_expected_field_count']} / {structural_results['total_rows_in_sample']}")
    print(f"DATE field valid/invalid: {structural_results['date_field_valid_rows']} / "
          f"{structural_results['date_field_invalid_rows']}")
    print(f"Non-empty content fields (of rows matching expected layout): "
          f"{structural_results['content_field_nonempty_row_counts']}")
    print(f"\nFull report written to:\n  {out_path}")


if __name__ == "__main__":
    main()