"""
Phase F (pre-check) — Data quality profile of the cleaned datasets

Read-only profiling of data/cleaned/. Prints summary tables in the
terminal and saves results/data_quality/data_quality_report.json.

Run (from any folder):
    python scripts/08_show_data_quality.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.paths import CLEANED_DIR, DATA_QUALITY_DIR, ensure_dirs


# =========================================================
# PROJECT PATHS
# =========================================================

ensure_dirs()

CLEANED = CLEANED_DIR
OUTPUT = DATA_QUALITY_DIR


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_value(value):
    """Convert pandas/numpy values into JSON-safe values."""
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


def dataframe_to_records(df):
    """Convert DataFrame into JSON-safe list of dictionaries."""
    records = []

    for record in df.to_dict(orient="records"):
        records.append({
            key: clean_value(value)
            for key, value in record.items()
        })

    return records


def print_table(title, df):
    """Display a clean table in VS Code terminal."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df.to_string(index=False))


# =========================================================
# LOAD FRESHRETAILNET
# =========================================================

train = pd.read_parquet(
    CLEANED / "freshretailnet_train.parquet"
)

eval_df = pd.read_parquet(
    CLEANED / "freshretailnet_eval.parquet"
)


# =========================================================
# FRESHRETAILNET SUMMARY
# =========================================================

fresh_summary = pd.DataFrame([
    [
        "Train",
        len(train),
        len(train.columns),
        int(train.isna().sum().sum()),
        int(
            train.duplicated(
                ["city_id", "store_id", "product_id", "dt"]
            ).sum()
        ),
        train["dt"].min(),
        train["dt"].max(),
    ],
    [
        "Eval",
        len(eval_df),
        len(eval_df.columns),
        int(eval_df.isna().sum().sum()),
        int(
            eval_df.duplicated(
                ["city_id", "store_id", "product_id", "dt"]
            ).sum()
        ),
        eval_df["dt"].min(),
        eval_df["dt"].max(),
    ],
], columns=[
    "Dataset",
    "Rows",
    "Columns",
    "Null Values",
    "Duplicate Keys",
    "Min Date",
    "Max Date",
])


# =========================================================
# FRESHRETAILNET COLUMN PROFILE
# =========================================================

fresh_profile = pd.DataFrame({
    "Column": train.columns,
    "Data Type": [
        str(train[c].dtype)
        for c in train.columns
    ],
    "Non Null": [
        int(train[c].notna().sum())
        for c in train.columns
    ],
    "Null Count": [
        int(train[c].isna().sum())
        for c in train.columns
    ],
    "Unique Values": [
        (
            "24-value array"
            if c in ["hours_sale", "hours_stock_status"]
            else int(train[c].nunique())
        )
        for c in train.columns
    ],
})


# =========================================================
# FRESHRETAILNET VALIDATION
# =========================================================

fresh_validation = pd.DataFrame([
    [
        "Train",
        "Missing values",
        int(train.isna().sum().sum())
    ],
    [
        "Eval",
        "Missing values",
        int(eval_df.isna().sum().sum())
    ],
    [
        "Train",
        "Duplicate composite keys",
        int(
            train.duplicated(
                ["city_id", "store_id", "product_id", "dt"]
            ).sum()
        )
    ],
    [
        "Eval",
        "Duplicate composite keys",
        int(
            eval_df.duplicated(
                ["city_id", "store_id", "product_id", "dt"]
            ).sum()
        )
    ],
    [
        "Train",
        "Invalid array lengths",
        int(
            (train["hours_sale"].map(len) != 24).sum()
        )
    ],
    [
        "Eval",
        "Invalid array lengths",
        int(
            (eval_df["hours_sale"].map(len) != 24).sum()
        )
    ],
    [
        "Train",
        "Discount > 1.0",
        int((train["discount"] > 1).sum())
    ],
    [
        "Eval",
        "Discount > 1.0",
        int((eval_df["discount"] > 1).sum())
    ],
    [
        "Train",
        "Discount == 0.0 (flagged, likely missing)",
        int((train["discount"] == 0).sum())
    ],
    [
        "Eval",
        "Discount == 0.0 (flagged, likely missing)",
        int((eval_df["discount"] == 0).sum())
    ],
    [
        "Train",
        "Stockout hours != sum of hourly flags (6-22)",
        int(
            (train["hours_stock_status"].map(lambda a: int(sum(a[6:22])))
             != train["stock_hour6_22_cnt"]).sum()
        )
    ],
    [
        "Train",
        "Store-product-days with at least one stockout hour",
        int((train["stock_hour6_22_cnt"] > 0).sum())
    ],
    [
        "Train / Eval",
        "Composite keys present in both splits",
        int(
            train[["city_id", "store_id", "product_id", "dt"]]
            .merge(eval_df[["city_id", "store_id", "product_id", "dt"]])
            .shape[0]
        )
    ],
], columns=[
    "Dataset",
    "Check",
    "Result"
])


# =========================================================
# LOAD GDELT
# =========================================================

gdelt = pd.read_parquet(
    CLEANED / "gdelt_cleaned.parquet"
)


# =========================================================
# GDELT SUMMARY
# =========================================================

gdelt_summary = pd.DataFrame([
    [
        "GDELT Cleaned",
        len(gdelt),
        len(gdelt.columns),
        int(gdelt.isna().sum().sum()),
        int(gdelt["gkg_record_id"].duplicated().sum()),
        gdelt["gkg_datetime_utc"].min(),
        gdelt["gkg_datetime_utc"].max(),
    ]
], columns=[
    "Dataset",
    "Rows",
    "Columns",
    "Null Values",
    "Duplicate GKG IDs",
    "Min Date",
    "Max Date",
])


# =========================================================
# GDELT COLUMN PROFILE
# =========================================================

gdelt_profile = pd.DataFrame({
    "Column": gdelt.columns,
    "Data Type": [
        str(gdelt[c].dtype)
        for c in gdelt.columns
    ],
    "Non Null": [
        int(gdelt[c].notna().sum())
        for c in gdelt.columns
    ],
    "Null Count": [
        int(gdelt[c].isna().sum())
        for c in gdelt.columns
    ],
    "Unique Values": [
        int(gdelt[c].nunique())
        for c in gdelt.columns
    ],
})


# =========================================================
# GDELT VALIDATION
# =========================================================

source_collection_numeric = pd.to_numeric(
    gdelt["source_collection_id"],
    errors="coerce"
)

tone_columns = [
    "tone_average",
    "tone_positive",
    "tone_negative",
    "tone_polarity",
    "tone_activity_reference_density",
    "tone_self_group_reference_density",
    "tone_word_count",
]

gdelt_validation = pd.DataFrame([
    [
        "Missing values",
        int(gdelt.isna().sum().sum())
    ],
    [
        "Duplicate GKG IDs",
        int(gdelt["gkg_record_id"].duplicated().sum())
    ],
    [
        "Invalid datetime values",
        int(gdelt["gkg_datetime_utc"].isna().sum())
    ],
    [
        "Invalid V2Tone rows",
        int(
            gdelt[tone_columns]
            .isna()
            .any(axis=1)
            .sum()
        )
    ],
    [
        "V2Locations structurally flagged",
        int(
            (gdelt["v2locations_structurally_valid"] == False)
            .sum()
        )
    ],
    [
        "Source collection IDs outside 1-6",
        int(
            (~source_collection_numeric.between(1, 6))
            .sum()
        )
    ],
], columns=[
    "Check",
    "Result"
])


# =========================================================
# CREATE JSON REPORT
# =========================================================

report = {
    "project": "retail_agentic_replenishment",
    "purpose": "Data quality profiling of cleaned datasets",

    "freshretailnet": {
        "summary": dataframe_to_records(fresh_summary),
        "column_profile": dataframe_to_records(fresh_profile),
        "validation_checks": dataframe_to_records(fresh_validation),
    },

    "gdelt": {
        "summary": dataframe_to_records(gdelt_summary),
        "column_profile": dataframe_to_records(gdelt_profile),
        "validation_checks": dataframe_to_records(gdelt_validation),
    },
}


# =========================================================
# SAVE JSON
# =========================================================

output_file = OUTPUT / "data_quality_report.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )


# =========================================================
# SHOW RESULTS IN VS CODE
# =========================================================

print_table(
    "FRESHRETAILNET — SUMMARY",
    fresh_summary
)

print_table(
    "FRESHRETAILNET — VALIDATION CHECKS",
    fresh_validation
)

print_table(
    "GDELT — SUMMARY",
    gdelt_summary
)

print_table(
    "GDELT — VALIDATION CHECKS",
    gdelt_validation
)

print("\n" + "=" * 100)
print("DATA QUALITY REPORT SAVED")
print("=" * 100)
print(output_file)