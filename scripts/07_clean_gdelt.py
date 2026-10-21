"""
Phase E — Cleaning & standardisation: GDELT 2.1 GKG

Implements the locked Phase E Cleaning Contract exactly:

- All 27 raw GKG source fields are represented in the cleaned output.
- DATE is preserved as raw text and parsed into a datetime column.
- SourceCollectionIdentifier is validated against documented values {1..6}.
- GKGRECORDID is validated as non-empty and corpus-unique.
- V2Tone is decomposed into its 7 documented numeric components.
- V2Themes / V2Locations / V2Persons / V2Organizations are preserved
  as raw strings. They are not analytically extracted or selected here.
- V1 counterparts are preserved as raw strings.
- Other raw GKG fields are preserved as raw strings.
- v2locations_structurally_valid flags rows containing the previously
  confirmed V2Locations field-count anomaly.
- Raw values are never modified.
- Full-corpus validation completes before the cleaned Parquet is written.
- No daily aggregation.
- No analytical feature selection.
- No row filtering.
- No raw data modification.

Run:
    python scripts/07_clean_gdelt.py
"""

import csv
import io
import json
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.paths import (
    RAW_GDELT_DIR,
    CLEANED_DIR,
    VALIDATION_DIR,
    INSPECTION_DIR,
    ensure_dirs,
)


# ---------------------------------------------------------------------
# GDELT 2.1 GKG source schema
# ---------------------------------------------------------------------

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

IDX = {
    name: i
    for i, name in enumerate(GKG_COLUMNS)
}


# ---------------------------------------------------------------------
# Documented validation domains
# ---------------------------------------------------------------------

VALID_SOURCE_COLLECTION_IDS = {
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
}

VALID_LOCATION_TYPES = {
    "1",
    "2",
    "3",
    "4",
    "5",
}


# ---------------------------------------------------------------------
# V2Tone output columns
# ---------------------------------------------------------------------

TONE_COLUMN_NAMES = [
    "tone_average",
    "tone_positive",
    "tone_negative",
    "tone_polarity",
    "tone_activity_reference_density",
    "tone_self_group_reference_density",
    "tone_word_count",
]


# ---------------------------------------------------------------------
# Direct raw-field pass-through mapping
# ---------------------------------------------------------------------

PASSTHROUGH_MAP = {
    "SourceCollectionIdentifier": "source_collection_id",
    "SourceCommonName": "source_common_name",
    "DocumentIdentifier": "document_identifier",
    "Counts": "v1_counts_raw",
    "V2Counts": "v2_counts_raw",
    "Themes": "v1_themes_raw",
    "V2Themes": "v2_themes_raw",
    "Locations": "v1_locations_raw",
    "V2Locations": "v2_locations_raw",
    "Persons": "v1_persons_raw",
    "V2Persons": "v2_persons_raw",
    "Organizations": "v1_organizations_raw",
    "V2Organizations": "v2_organizations_raw",
    "Dates": "v2_enhanced_dates_raw",
    "GCAM": "v2_gcam_raw",
    "SharingImage": "sharing_image_url",
    "RelatedImages": "related_images_raw",
    "SocialImageEmbeds": "social_image_embeds_raw",
    "SocialVideoEmbeds": "social_video_embeds_raw",
    "Quotations": "quotations_raw",
    "AllNames": "all_names_raw",
    "Amounts": "amounts_raw",
    "TranslationInfo": "translation_info_raw",
    "Extras": "extras_raw",
}


# ---------------------------------------------------------------------
# Expected cleaned schema
# ---------------------------------------------------------------------

EXPECTED_OUTPUT_COLUMNS = (
    [
        "gkg_record_id",
        "gkg_date_raw",
        "gkg_datetime_utc",
    ]
    + list(PASSTHROUGH_MAP.values())
    + TONE_COLUMN_NAMES
    + [
        "v2locations_structurally_valid",
    ]
)

EXPECTED_OUTPUT_COLUMN_COUNT = len(EXPECTED_OUTPUT_COLUMNS)  # 35


# ---------------------------------------------------------------------
# Custom validation exception
# ---------------------------------------------------------------------

class ValidationFailure(Exception):
    pass


# ---------------------------------------------------------------------
# Streaming GDELT reader
# ---------------------------------------------------------------------

def read_zip_rows_streaming(fp):
    """
    Stream rows from the first member of a GDELT GKG ZIP archive.

    The CSV field-size limit is increased because GDELT compound fields
    can be much larger than Python's default CSV field limit.
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

            reader = csv.reader(
                text,
                delimiter="\t",
            )

            for row in reader:
                yield row


# ---------------------------------------------------------------------
# V2Locations structural validation
# ---------------------------------------------------------------------

def validate_locations_field(value):
    """
    Return True when the V2Locations field contains no structural
    field-count anomaly.

    According to the locked Phase E diagnostic:

    - empty V2Locations is valid;
    - each populated block should contain 9 '#' separated fields;
    - the confirmed anomaly is field-count deviation;
    - Location Type validation is not used here because the diagnostic
      confirmed zero type-only failures.

    Raw values are never modified.
    """

    if value == "":
        return True

    for block in value.split(";"):

        if block == "":
            continue

        parts = block.split("#")

        if len(parts) != 9:
            return False

    return True


# ---------------------------------------------------------------------
# V2Tone parsing
# ---------------------------------------------------------------------

def parse_tone(value):
    """
    Parse V2Tone into its seven documented numeric components.

    Returns:
        list of 7 floats if valid
        None if structurally invalid
    """

    if value == "":
        return None

    parts = value.split(",")

    if len(parts) != 7:
        return None

    try:
        return [
            float(part)
            for part in parts
        ]

    except ValueError:
        return None


# ---------------------------------------------------------------------
# Main cleaning process
# ---------------------------------------------------------------------

def main():

    ensure_dirs()

    # -------------------------------------------------------------
    # Increase CSV field-size limit safely
    # -------------------------------------------------------------

    max_field_size = sys.maxsize

    while True:

        try:

            csv.field_size_limit(max_field_size)

            break

        except OverflowError:

            max_field_size //= 10

    # -------------------------------------------------------------
    # Locate raw GDELT files
    # -------------------------------------------------------------

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
        f"Phase E cleaning: streaming "
        f"{len(all_files)} raw GDELT files, full corpus...\n"
    )

    # -------------------------------------------------------------
    # In-memory records
    # -------------------------------------------------------------

    records = []

    # -------------------------------------------------------------
    # Corpus-level counters
    # -------------------------------------------------------------

    total_rows = 0

    gkgid_seen = set()

    field_count_failures = 0

    gkgid_empty_or_dup = 0

    date_failures = 0

    source_collection_failures = 0

    source_collection_value_counts = Counter()

    tone_failures = 0

    v2locations_invalid_count = 0

    # -------------------------------------------------------------
    # Process every raw GDELT file
    # -------------------------------------------------------------

    for file_idx, fp in enumerate(
        all_files,
        start=1,
    ):

        file_rel = str(
            fp.relative_to(RAW_GDELT_DIR)
        )

        try:

            row_iter = read_zip_rows_streaming(fp)

            for row_num, row in enumerate(
                row_iter,
                start=1,
            ):

                total_rows += 1

                # -------------------------------------------------
                # 1. Field count
                # -------------------------------------------------

                if len(row) != EXPECTED_FIELD_COUNT:

                    field_count_failures += 1

                    raise ValidationFailure(
                        f"File '{file_rel}' row {row_num}: "
                        f"field count {len(row)} != "
                        f"{EXPECTED_FIELD_COUNT}"
                    )

                # -------------------------------------------------
                # 2. GKGRECORDID
                # -------------------------------------------------

                gkgid = row[
                    IDX["GKGRECORDID"]
                ]

                if (
                    gkgid == ""
                    or gkgid in gkgid_seen
                ):

                    gkgid_empty_or_dup += 1

                    raise ValidationFailure(
                        f"File '{file_rel}' row {row_num}: "
                        f"GKGRECORDID empty or duplicate "
                        f"('{gkgid}')"
                    )

                gkgid_seen.add(gkgid)

                # -------------------------------------------------
                # 3. DATE
                # -------------------------------------------------

                date_raw = row[
                    IDX["DATE"]
                ]

                if not (
                    len(date_raw) == 14
                    and date_raw.isdigit()
                ):

                    date_failures += 1

                    raise ValidationFailure(
                        f"File '{file_rel}' row {row_num}: "
                        f"DATE '{date_raw}' not a valid "
                        f"14-digit timestamp"
                    )

                try:

                    dt_utc = datetime.strptime(
                        date_raw,
                        "%Y%m%d%H%M%S",
                    ).replace(
                        tzinfo=timezone.utc
                    )

                except ValueError:

                    date_failures += 1

                    raise ValidationFailure(
                        f"File '{file_rel}' row {row_num}: "
                        f"DATE '{date_raw}' cannot be parsed"
                    )

                # -------------------------------------------------
                # 4. SourceCollectionIdentifier
                # -------------------------------------------------

                sci = row[
                    IDX["SourceCollectionIdentifier"]
                ]

                source_collection_value_counts[
                    sci
                ] += 1

                if sci not in VALID_SOURCE_COLLECTION_IDS:

                    source_collection_failures += 1

                    raise ValidationFailure(
                        f"File '{file_rel}' row {row_num}: "
                        f"SourceCollectionIdentifier "
                        f"'{sci}' not in documented {{1..6}}"
                    )

                # -------------------------------------------------
                # 5. V2Tone
                # -------------------------------------------------

                tone_parts = parse_tone(
                    row[IDX["V2Tone"]]
                )

                if tone_parts is None:

                    tone_failures += 1

                    raise ValidationFailure(
                        f"File '{file_rel}' row {row_num}: "
                        f"V2Tone structurally invalid"
                    )

                # -------------------------------------------------
                # 6. V2Locations
                # -------------------------------------------------

                loc_valid = validate_locations_field(
                    row[IDX["V2Locations"]]
                )

                if not loc_valid:

                    # This is NOT a hard failure.
                    #
                    # The anomaly was previously diagnosed across
                    # the full corpus and is explicitly retained
                    # and flagged under the locked Phase E contract.

                    v2locations_invalid_count += 1

                # -------------------------------------------------
                # 7. Build cleaned record
                # -------------------------------------------------

                record = {
                    "gkg_record_id": gkgid,
                    "gkg_date_raw": date_raw,
                    "gkg_datetime_utc": dt_utc,
                }

                # Direct raw pass-through fields

                for raw_col, clean_col in PASSTHROUGH_MAP.items():

                    record[
                        clean_col
                    ] = row[
                        IDX[raw_col]
                    ]

                # V2Tone components

                for name, value in zip(
                    TONE_COLUMN_NAMES,
                    tone_parts,
                ):

                    record[name] = value

                # V2Locations structural flag

                record[
                    "v2locations_structurally_valid"
                ] = loc_valid

                records.append(record)

        except ValidationFailure as e:

            print(
                f"\n[VALIDATION FAILED] {e}"
            )

            print(
                "No cleaned parquet file was written. "
                "Raw data is untouched."
            )

            fail_report = {
                "generated_at_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "overall_status":
                    "FAIL",

                "error":
                    str(e),

                "rows_processed_before_failure":
                    total_rows,
            }

            out_path = (
                VALIDATION_DIR
                / "gdelt_cleaning_validation.json"
            )

            with open(
                out_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    fail_report,
                    f,
                    indent=2,
                    default=str,
                )

            print(
                f"Failure report written to: "
                f"{out_path}"
            )

            sys.exit(1)

        # ---------------------------------------------------------
        # Progress reporting
        # ---------------------------------------------------------

        if (
            file_idx % 50 == 0
            or file_idx == len(all_files)
        ):

            print(
                f"  Processed "
                f"{file_idx}/{len(all_files)} files, "
                f"{total_rows} rows so far..."
            )

    # -------------------------------------------------------------
    # All hard validations passed
    # -------------------------------------------------------------

    print(
        f"\nAll {total_rows} rows passed "
        f"hard validation. Building output DataFrame..."
    )

    df = pd.DataFrame.from_records(
        records,
        columns=EXPECTED_OUTPUT_COLUMNS,
    )

    # -------------------------------------------------------------
    # Final structural checks
    # -------------------------------------------------------------

    if df.shape[0] != total_rows:

        print(
            f"[VALIDATION FAILED] Output row count "
            f"{df.shape[0]} != input row count "
            f"{total_rows}"
        )

        sys.exit(1)

    if df.shape[1] != EXPECTED_OUTPUT_COLUMN_COUNT:

        print(
            f"[VALIDATION FAILED] Output column count "
            f"{df.shape[1]} != expected "
            f"{EXPECTED_OUTPUT_COLUMN_COUNT}"
        )

        sys.exit(1)

    if (
        df["gkg_record_id"].nunique()
        != total_rows
    ):

        print(
            "[VALIDATION FAILED] "
            "gkg_record_id not unique "
            "in final output"
        )

        sys.exit(1)

    # -------------------------------------------------------------
    # Write cleaned output
    # -------------------------------------------------------------

    out_path = (
        CLEANED_DIR
        / "gdelt_cleaned.parquet"
    )

    df.to_parquet(
        out_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Calculate anomaly percentage
    # -------------------------------------------------------------

    v2locations_affected_pct = (
        v2locations_invalid_count
        / total_rows
    ) * 100

    # Malformed-block total and failure-mode split come from the
    # 06b diagnostic report (measured, not typed in by hand).

    diag_path = (
        INSPECTION_DIR
        / "gdelt_v2locations_failure_modes_report.json"
    )

    if diag_path.exists():

        with open(diag_path, encoding="utf-8") as f:
            diag = json.load(f)

        malformed_blocks_text = (
            f"{diag['total_malformed_blocks']:,} malformed blocks in total, "
            f"measured by scripts/06b_diagnose_v2locations.py"
        )

        type_only_text = (
            f"{diag['rows_failing_location_type_only']} rows failed only "
            "on invalid Location Type values among nine-field blocks"
        )

    else:

        malformed_blocks_text = (
            "malformed block total not available; "
            "run scripts/06b_diagnose_v2locations.py"
        )

        type_only_text = (
            "Location Type failure split not available"
        )

    # -------------------------------------------------------------
    # Validation report
    # -------------------------------------------------------------

    report = {

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "overall_status":
            "PASS",

        "input": {

            "total_files":
                len(all_files),

            "total_rows":
                total_rows,
        },

        "output": {

            "path":
                str(out_path),

            "rows":
                int(df.shape[0]),

            "columns":
                int(df.shape[1]),

            "column_list":
                list(df.columns),
        },

        "validation": {

            "field_count_27_check": {

                "hard_fail_count":
                    field_count_failures,
            },

            "gkgrecordid_check": {

                "hard_fail_count":
                    gkgid_empty_or_dup,

                "unique_count":
                    len(gkgid_seen),
            },

            "date_check": {

                "hard_fail_count":
                    date_failures,
            },

            "source_collection_identifier_check": {

                "hard_fail_count":
                    source_collection_failures,

                "observed_value_counts":
                    dict(
                        source_collection_value_counts
                    ),
            },

            "v2tone_check": {

                "hard_fail_count":
                    tone_failures,
            },
        },

        "v2locations_anomaly": {

            "affected_rows":
                v2locations_invalid_count,

            "affected_row_percentage":
                round(
                    v2locations_affected_pct,
                    4,
                ),

            "total_rows":
                total_rows,

            "note":
                (
                    f"{v2locations_invalid_count:,} of {total_rows:,} GDELT records "
                    f"({v2locations_affected_pct:.4f}%) contain one or more "
                    "structurally non-conforming "
                    "V2Locations blocks "
                    f"({malformed_blocks_text}). "
                    "All flagged rows are due to "
                    "field-count deviations from the "
                    "documented nine-field structure; "
                    f"{type_only_text}. Raw values are "
                    "retained unmodified for all affected "
                    "rows. The underlying mechanism producing "
                    "this pattern is not asserted."
                ),
        },

        "scope_notes": {

            "fields_preserved":
                (
                    "All 27 raw GKG source fields "
                    "are represented in the cleaned "
                    "output; no source field is silently "
                    "discarded."
                ),

            "v2tone_transformation":
                (
                    "V2Tone is decomposed into its seven "
                    "documented numeric components."
                ),

            "date_transformation":
                (
                    "DATE is preserved as gkg_date_raw "
                    "and additionally parsed as "
                    "gkg_datetime_utc."
                ),

            "fields_structurally_validated": [

                "GKGRECORDID",

                "DATE",

                "SourceCollectionIdentifier",

                "V2Tone",

                "V2Themes "
                "(corpus-wide, prior script)",

                "V2Locations "
                "(corpus-wide, prior script)",

                "V2Persons "
                "(corpus-wide, prior script)",

                "V2Organizations "
                "(corpus-wide, prior script)",
            ],

            "daily_aggregation":
                (
                    "NOT performed — "
                    "deferred to Phase F."
                ),

            "analytical_feature_selection":
                (
                    "NOT performed — "
                    "deferred to Phase F."
                ),

            "row_filtering":
                "NOT performed.",

            "raw_data_modification":
                "NOT performed.",
        },
    }

    # -------------------------------------------------------------
    # Write validation report
    # -------------------------------------------------------------

    report_path = (
        VALIDATION_DIR
        / "gdelt_cleaning_validation.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            default=str,
        )

    # -------------------------------------------------------------
    # Final console summary
    # -------------------------------------------------------------

    print(
        f"\nCleaned file written:\n"
        f"  {out_path}"
    )

    print(
        f"Shape: "
        f"{df.shape[0]} rows x "
        f"{df.shape[1]} columns"
    )

    print(
        f"Validation report written to:\n"
        f"  {report_path}"
    )

    print(
        f"\nV2Locations anomaly: "
        f"{v2locations_invalid_count} rows "
        f"({v2locations_affected_pct:.4f}%) "
        f"flagged, all preserved."
    )

    print(
        "SourceCollectionIdentifier distribution: "
        f"{dict(source_collection_value_counts)}"
    )

    print(
        "\nOverall status: PASS"
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()