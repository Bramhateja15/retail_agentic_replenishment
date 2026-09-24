"""
Phase E, Step 1 — Full-corpus GDELT validation (READ ONLY)

Validates 100% of the raw GDELT GKG corpus (all 412 files, every row)
against structural rules confirmed from the official GDELT 2.1 GKG
codebook (http://data.gdeltproject.org/documentation/
GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf).

This is validation-only:
- No cleaned files are written.
- No malformed-row treatment decision is made here.
- Raw data is never modified.

Output:
    results/inspection/gdelt_full_corpus_validation.json

Run:
    python scripts/06_validate_gdelt_corpus.py
"""

import csv
import io
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import RAW_GDELT_DIR, INSPECTION_DIR, ensure_dirs


GKG_COLUMNS = [
    "GKGRECORDID",
    "DATE",
    "SourceCollectionIdentifier",
    "SourceCommonName",
    "DocumentIdentifier",
    "Counts",
    "V2Counts",
    "Themes",
    "V2Themes",
    "Locations",
    "V2Locations",
    "Persons",
    "V2Persons",
    "Organizations",
    "V2Organizations",
    "V2Tone",
    "Dates",
    "GCAM",
    "SharingImage",
    "RelatedImages",
    "SocialImageEmbeds",
    "SocialVideoEmbeds",
    "Quotations",
    "AllNames",
    "Amounts",
    "TranslationInfo",
    "Extras",
]

EXPECTED_FIELD_COUNT = len(GKG_COLUMNS)  # 27
IDX = {name: i for i, name in enumerate(GKG_COLUMNS)}

MAX_EXAMPLES_PER_CHECK = 15

VALID_LOCATION_TYPES = {"1", "2", "3", "4", "5"}


def new_example_store():
    return {"count": 0, "examples": []}


def record_failure(
    store,
    file_rel,
    row_num,
    gkgid,
    field,
    raw_value,
    reason,
):
    store["count"] += 1

    if len(store["examples"]) < MAX_EXAMPLES_PER_CHECK:
        store["examples"].append(
            {
                "file": file_rel,
                "row_number": row_num,
                "gkgrecordid": gkgid,
                "field": field,
                "raw_value": (
                    raw_value[:200] + "..."
                    if len(raw_value) > 200
                    else raw_value
                ),
                "failure_reason": reason,
            }
        )


def validate_theme_person_org_field(value):
    """
    ';'-separated 'text,offset' pairs.

    Empty string is valid.
    Split each entry on the LAST comma because text may itself contain commas.
    """
    if value == "":
        return True, None

    entries = value.split(";")

    for entry in entries:
        if entry == "":
            continue  # trailing semicolon artifact; not itself an error

        if "," not in entry:
            return (
                False,
                f"entry missing comma-separated offset: '{entry[:60]}'",
            )

        text_part, _, offset_part = entry.rpartition(",")

        if text_part == "":
            return (
                False,
                f"entry has empty text before offset: '{entry[:60]}'",
            )

        if not offset_part.isdigit():
            return (
                False,
                f"offset not a non-negative integer: '{offset_part[:30]}'",
            )

    return True, None


def validate_locations_field(value):
    """
    ';'-separated blocks of 9 '#'-delimited fields.

    Field 1 (index 0) must be in {1,2,3,4,5}.
    Empty string is valid.
    """
    if value == "":
        return True, None

    blocks = value.split(";")

    for block in blocks:
        if block == "":
            continue

        parts = block.split("#")

        if len(parts) != 9:
            return (
                False,
                f"expected 9 '#'-delimited fields, got {len(parts)}: "
                f"'{block[:80]}'",
            )

        if parts[0] not in VALID_LOCATION_TYPES:
            return (
                False,
                f"Location Type '{parts[0]}' not in documented set "
                f"{{1,2,3,4,5}}",
            )

    return True, None


def validate_tone_field(value):
    """
    Exactly 7 comma-separated floats.

    Empty string is tracked separately and is not necessarily a failure.
    """
    if value == "":
        return True, None, True

    parts = value.split(",")

    if len(parts) != 7:
        return (
            False,
            f"expected 7 comma-separated values, got {len(parts)}: "
            f"'{value[:80]}'",
            False,
        )

    for p in parts:
        try:
            float(p)
        except ValueError:
            return (
                False,
                f"non-numeric tone component: '{p[:30]}'",
                False,
            )

    return True, None, False


def validate_date_field(value):
    """
    14-digit YYYYMMDDHHMMSS,
    OR documented special value '0'.
    """
    if value == "0":
        return "zero_special_case"

    if len(value) == 14 and value.isdigit():
        return "valid"

    return "invalid"

def read_zip_rows_streaming(fp):
    """
    Generator: yields rows one at a time.

    The full ZIP member is never materialized into a list.
    """
    with zipfile.ZipFile(fp) as z:
        names = z.namelist()

        if not names:
            return

        with z.open(names[0]) as raw:
            text = io.TextIOWrapper(
                raw,
                encoding="utf-8",
                errors="replace",
            )
            # GDELT contains very large compound text fields.
            # Increase Python's CSV field-size limit so valid long
            # GDELT fields can be read without parser failure.
            max_field_size = sys.maxsize
            while True:
                try:
                    csv.field_size_limit(max_field_size)
                    break
                except OverflowError:
                    max_field_size //= 10
            reader = csv.reader(text, delimiter="\t")
            for row in reader:
                yield row


def main():
    ensure_dirs()

    all_files = sorted(
        RAW_GDELT_DIR.rglob("*.gkg.csv.zip")
    )

    if not all_files:
        print(
            f"[STOP] No raw GDELT files found under "
            f"{RAW_GDELT_DIR}."
        )
        print(
            "Run scripts/02_download_gdelt.py first."
        )
        sys.exit(1)

    print(
        f"Validating FULL GDELT corpus: "
        f"{len(all_files)} files, every row (no sampling)..."
    )
    print("This may take several minutes.\n")

    total_files = len(all_files)
    total_rows = 0

    rows_by_file = {}

    files_with_failures = set()

    field_count_failures = new_example_store()

    date_valid = 0
    date_zero_special = 0
    date_invalid = new_example_store()

    gkgid_seen = set()
    gkgid_duplicates = new_example_store()
    gkgid_empty = new_example_store()

    tone_structural_failures = new_example_store()
    tone_empty_count = 0
    tone_out_of_documented_range = new_example_store()

    themes_failures = new_example_store()
    themes_empty_count = 0

    locations_failures = new_example_store()
    locations_empty_count = 0

    persons_failures = new_example_store()
    persons_empty_count = 0

    orgs_failures = new_example_store()
    orgs_empty_count = 0

    zip_open_failures = new_example_store()

    for file_idx, fp in enumerate(all_files, start=1):
        file_rel = str(
            fp.relative_to(RAW_GDELT_DIR)
        )

        file_had_failure = False
        file_row_count = 0

        try:
            row_iter = read_zip_rows_streaming(fp)

            for row_num, row in enumerate(
                row_iter,
                start=1,
            ):
                file_row_count += 1
                total_rows += 1

                # --------------------------------------------------
                # Field-count validation
                # --------------------------------------------------
                if len(row) != EXPECTED_FIELD_COUNT:
                    record_failure(
                        field_count_failures,
                        file_rel,
                        row_num,
                        None,
                        "row",
                        f"{len(row)} fields",
                        "field count != 27",
                    )

                    file_had_failure = True
                    continue

                # --------------------------------------------------
                # GKGRECORDID validation
                # --------------------------------------------------
                gkgid = row[IDX["GKGRECORDID"]]

                if gkgid == "":
                    record_failure(
                        gkgid_empty,
                        file_rel,
                        row_num,
                        None,
                        "GKGRECORDID",
                        "",
                        "empty record id",
                    )

                    file_had_failure = True

                elif gkgid in gkgid_seen:
                    record_failure(
                        gkgid_duplicates,
                        file_rel,
                        row_num,
                        gkgid,
                        "GKGRECORDID",
                        gkgid,
                        "duplicate across corpus",
                    )

                    file_had_failure = True

                else:
                    gkgid_seen.add(gkgid)

                # --------------------------------------------------
                # DATE validation
                # --------------------------------------------------
                date_status = validate_date_field(
                    row[IDX["DATE"]]
                )

                if date_status == "valid":
                    date_valid += 1

                elif date_status == "zero_special_case":
                    date_zero_special += 1

                else:
                    record_failure(
                        date_invalid,
                        file_rel,
                        row_num,
                        gkgid,
                        "DATE",
                        row[IDX["DATE"]],
                        "not 14-digit and not documented '0'",
                    )

                    file_had_failure = True

                # --------------------------------------------------
                # V2Tone validation
                # --------------------------------------------------
                tone_ok, tone_reason, tone_is_empty = (
                    validate_tone_field(
                        row[IDX["V2Tone"]]
                    )
                )

                if tone_is_empty:
                    tone_empty_count += 1

                elif not tone_ok:
                    record_failure(
                        tone_structural_failures,
                        file_rel,
                        row_num,
                        gkgid,
                        "V2Tone",
                        row[IDX["V2Tone"]],
                        tone_reason,
                    )

                    file_had_failure = True

                else:
                    tone_val = float(
                        row[IDX["V2Tone"]].split(",")[0]
                    )

                    if not (-100 <= tone_val <= 100):
                        record_failure(
                            tone_out_of_documented_range,
                            file_rel,
                            row_num,
                            gkgid,
                            "V2Tone",
                            row[IDX["V2Tone"]],
                            (
                                f"Tone component {tone_val} "
                                "outside documented [-100,100]"
                            ),
                        )

                # --------------------------------------------------
                # V2Themes validation
                # --------------------------------------------------
                themes_val = row[IDX["V2Themes"]]

                if themes_val == "":
                    themes_empty_count += 1

                else:
                    ok, reason = (
                        validate_theme_person_org_field(
                            themes_val
                        )
                    )

                    if not ok:
                        record_failure(
                            themes_failures,
                            file_rel,
                            row_num,
                            gkgid,
                            "V2Themes",
                            themes_val,
                            reason,
                        )

                        file_had_failure = True

                # --------------------------------------------------
                # V2Locations validation
                # --------------------------------------------------
                loc_val = row[IDX["V2Locations"]]

                if loc_val == "":
                    locations_empty_count += 1

                else:
                    ok, reason = validate_locations_field(
                        loc_val
                    )

                    if not ok:
                        record_failure(
                            locations_failures,
                            file_rel,
                            row_num,
                            gkgid,
                            "V2Locations",
                            loc_val,
                            reason,
                        )

                        file_had_failure = True

                # --------------------------------------------------
                # V2Persons validation
                # --------------------------------------------------
                pers_val = row[IDX["V2Persons"]]

                if pers_val == "":
                    persons_empty_count += 1

                else:
                    ok, reason = (
                        validate_theme_person_org_field(
                            pers_val
                        )
                    )

                    if not ok:
                        record_failure(
                            persons_failures,
                            file_rel,
                            row_num,
                            gkgid,
                            "V2Persons",
                            pers_val,
                            reason,
                        )

                        file_had_failure = True

                # --------------------------------------------------
                # V2Organizations validation
                # --------------------------------------------------
                org_val = row[IDX["V2Organizations"]]

                if org_val == "":
                    orgs_empty_count += 1

                else:
                    ok, reason = (
                        validate_theme_person_org_field(
                            org_val
                        )
                    )

                    if not ok:
                        record_failure(
                            orgs_failures,
                            file_rel,
                            row_num,
                            gkgid,
                            "V2Organizations",
                            org_val,
                            reason,
                        )

                        file_had_failure = True

        except zipfile.BadZipFile as e:
            record_failure(
                zip_open_failures,
                file_rel,
                None,
                None,
                "zip",
                "",
                str(e),
            )

            files_with_failures.add(file_rel)
            rows_by_file[file_rel] = 0

            continue

        rows_by_file[file_rel] = file_row_count

        if file_had_failure:
            files_with_failures.add(file_rel)

        if (
            file_idx % 50 == 0
            or file_idx == total_files
        ):
            print(
                f"  Processed {file_idx}/{total_files} files, "
                f"{total_rows} rows so far..."
            )

    # --------------------------------------------------------------
    # Build validation report
    # --------------------------------------------------------------
    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "codebook_source": (
            "http://data.gdeltproject.org/documentation/"
            "GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf"
        ),

        "corpus_summary": {
            "total_files_examined": total_files,
            "total_rows_examined": total_rows,
            "files_with_any_failure": len(
                files_with_failures
            ),
            "files_with_failures_list": sorted(
                files_with_failures
            )[:50],
        },

        "rows_by_file_sample": dict(
            list(rows_by_file.items())[:20]
        ),

        "checks": {
            "zip_integrity": zip_open_failures,

            "field_count_27": field_count_failures,

            "gkgrecordid_empty": gkgid_empty,

            "gkgrecordid_duplicates": gkgid_duplicates,

            "gkgrecordid_unique_count": len(
                gkgid_seen
            ),

            "date_valid_count": date_valid,

            "date_zero_special_case_count": (
                date_zero_special
            ),

            "date_invalid": date_invalid,

            "v2tone_structural_failures": (
                tone_structural_failures
            ),

            "v2tone_empty_count": tone_empty_count,

            "v2tone_out_of_documented_range_examples": (
                tone_out_of_documented_range
            ),

            "v2themes_failures": themes_failures,

            "v2themes_empty_count": themes_empty_count,

            "v2locations_failures": locations_failures,

            "v2locations_empty_count": (
                locations_empty_count
            ),

            "v2persons_failures": persons_failures,

            "v2persons_empty_count": (
                persons_empty_count
            ),

            "v2organizations_failures": orgs_failures,

            "v2organizations_empty_count": (
                orgs_empty_count
            ),
        },
    }

    # --------------------------------------------------------------
    # Write validation report
    # --------------------------------------------------------------
    out_path = (
        INSPECTION_DIR
        / "gdelt_full_corpus_validation.json"
    )

    with open(out_path, "w") as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    # --------------------------------------------------------------
    # Console summary
    # --------------------------------------------------------------
    print(
        "\n=== FULL-CORPUS GDELT VALIDATION SUMMARY ==="
    )

    print(
        f"Files examined: {total_files} | "
        f"Rows examined: {total_rows}"
    )

    print(
        f"Files with any failure: "
        f"{len(files_with_failures)}"
    )

    print(
        f"Zip integrity failures: "
        f"{zip_open_failures['count']}"
    )

    print(
        f"Field-count (!=27) failures: "
        f"{field_count_failures['count']}"
    )

    print(
        f"GKGRECORDID empty: "
        f"{gkgid_empty['count']} | "
        f"duplicates: "
        f"{gkgid_duplicates['count']} | "
        f"unique: "
        f"{len(gkgid_seen)}"
    )

    print(
        f"DATE valid: {date_valid} | "
        f"zero-special-case: {date_zero_special} | "
        f"invalid: {date_invalid['count']}"
    )

    print(
        f"V2Tone structural failures: "
        f"{tone_structural_failures['count']} | "
        f"empty: {tone_empty_count} | "
        f"out-of-documented-range "
        f"(informational): "
        f"{tone_out_of_documented_range['count']}"
    )

    print(
        f"V2Themes failures: "
        f"{themes_failures['count']} | "
        f"empty: {themes_empty_count}"
    )

    print(
        f"V2Locations failures: "
        f"{locations_failures['count']} | "
        f"empty: {locations_empty_count}"
    )

    print(
        f"V2Persons failures: "
        f"{persons_failures['count']} | "
        f"empty: {persons_empty_count}"
    )

    print(
        f"V2Organizations failures: "
        f"{orgs_failures['count']} | "
        f"empty: {orgs_empty_count}"
    )

    print(
        f"\nFull report written to:\n  {out_path}"
    )


if __name__ == "__main__":
    main()