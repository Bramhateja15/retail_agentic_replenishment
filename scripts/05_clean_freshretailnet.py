"""
Phase D — Cleaning & standardisation: FreshRetailNet-50K

Implements the Phase D Cleaning Contract exactly:
  - dt: str -> datetime64[ns], hard failure on any unparseable value
  - hours_sale / hours_stock_status: kept as 24-element arrays, unchanged
  - discount: values UNCHANGED; new boolean flags discount_above_reference
    (discount > 1.0) and discount_zero (discount == 0.0, a 100% discount,
    most likely a missing price rather than a free product)
  - Full-dataset (not sampled) structural and cross-field validation
  - No imputation, capping, filtering, aggregation, or feature engineering
  - Row counts must be preserved exactly
  - Validation runs to completion BEFORE any cleaned file is written;
    if either split fails, NEITHER cleaned file is written.

Run:
    python scripts/05_clean_freshretailnet.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config.paths import RAW_FRESHRETAILNET_DIR, CLEANED_DIR, VALIDATION_DIR, ensure_dirs

EXPECTED_INPUT_COLUMNS = [
    "city_id", "store_id", "management_group_id", "first_category_id",
    "second_category_id", "third_category_id", "product_id", "dt",
    "sale_amount", "hours_sale", "stock_hour6_22_cnt", "hours_stock_status",
    "discount", "holiday_flag", "activity_flag", "precpt",
    "avg_temperature", "avg_humidity", "avg_wind_level",
]
KEY_COLUMNS = ["city_id", "store_id", "product_id", "dt"]
ARRAY_COLUMNS = ["hours_sale", "hours_stock_status"]
FLAG_COLUMNS = ["holiday_flag", "activity_flag"]
EXPECTED_SPLIT_ROWS = {"train": 4_500_000, "eval": 350_000}


class ValidationFailure(Exception):
    """Raised when a hard contract validation fails. Caught in main()
    to prevent any cleaned parquet from being written."""
    pass


def clean_and_validate_split(name, raw_path):
    """Returns (cleaned_df, validation_dict). Raises ValidationFailure
    on any hard-fail check. Never writes anything to disk."""

    result = {"split": name, "path": str(raw_path)}

    df = pd.read_parquet(raw_path)
    input_rows, input_cols = df.shape
    result["input_rows"] = int(input_rows)
    result["input_columns"] = int(input_cols)

    # --- schema presence check -------------------------------------
    missing_cols = set(EXPECTED_INPUT_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValidationFailure(f"[{name}] Missing expected columns: {missing_cols}")
    extra_cols = set(df.columns) - set(EXPECTED_INPUT_COLUMNS)
    result["unexpected_input_columns"] = sorted(extra_cols)

    # --- dt: parse, hard-fail on any unparseable value ----------------
    parsed_dt = pd.to_datetime(df["dt"], errors="coerce")
    n_unparseable = int(parsed_dt.isna().sum())
    if n_unparseable > 0:
        bad_rows = df.loc[parsed_dt.isna()].index.tolist()[:20]  # cap preview
        raise ValidationFailure(
            f"[{name}] {n_unparseable} rows failed dt parsing. "
            f"First offending row indices: {bad_rows}"
        )
    df = df.copy()
    df["dt"] = parsed_dt
    result["dtype_changes"] = {"dt": "str -> datetime64[ns]"}

    # --- missing values: must remain 0 everywhere -----------------
    missing_report = {c: int(df[c].isna().sum()) for c in df.columns}
    if any(v > 0 for v in missing_report.values()):
        raise ValidationFailure(f"[{name}] Unexpected missing values introduced: {missing_report}")
    result["missing_value_check"] = missing_report

    # --- composite key uniqueness ------------------------------------
    dup_count = int(df.duplicated(subset=KEY_COLUMNS).sum())
    if dup_count > 0:
        raise ValidationFailure(f"[{name}] {dup_count} duplicate composite-key rows found.")
    result["duplicate_key_check"] = {"key_columns": KEY_COLUMNS, "duplicate_count": dup_count}

    # --- array length validation (FULL dataset) ------------------------
    array_check = {}
    for c in ARRAY_COLUMNS:
        lengths = df[c].map(len)
        bad = int((lengths != 24).sum())
        if bad > 0:
            bad_idx = df.index[lengths != 24].tolist()[:20]
            raise ValidationFailure(
                f"[{name}] {bad} rows in '{c}' do not have length 24. "
                f"First offending row indices: {bad_idx}"
            )
        array_check[c] = {"all_length_24": True, "rows_checked": int(len(df))}
    result["array_length_check"] = array_check

    # --- hours_stock_status values must be binary --------------------
    all_vals = np.unique(np.concatenate(df["hours_stock_status"].values))
    if not set(all_vals.tolist()).issubset({0, 1, 0.0, 1.0}):
        raise ValidationFailure(f"[{name}] hours_stock_status contains non-binary values: {all_vals}")
    result["hours_stock_status_value_check"] = {"observed_values": sorted(float(v) for v in all_vals)}

    # --- cross-field consistency (FULL dataset, not sampled) -----------
    computed = df["hours_stock_status"].map(lambda a: int(np.sum(a[6:22])))
    mismatch_mask = computed != df["stock_hour6_22_cnt"]
    mismatch_count = int(mismatch_mask.sum())
    if mismatch_count > 0:
        bad_idx = df.index[mismatch_mask].tolist()[:20]
        raise ValidationFailure(
            f"[{name}] {mismatch_count} rows fail stock_hour6_22_cnt == "
            f"sum(hours_stock_status[6:22]). First offending row indices: {bad_idx}"
        )
    result["cross_field_check"] = {
        "check": "stock_hour6_22_cnt == sum(hours_stock_status[6:22])",
        "rows_checked": int(len(df)),
        "mismatch_count": mismatch_count,
    }

    # --- binary flag validation --------------------------------------
    flag_check = {}
    for c in FLAG_COLUMNS:
        vals = set(df[c].unique().tolist())
        if not vals.issubset({0, 1}):
            raise ValidationFailure(f"[{name}] {c} contains non-binary values: {vals}")
        flag_check[c] = sorted(vals)
    result["binary_flag_check"] = flag_check

    # --- non-negativity checks (documented bounds only) -----------------
    for c in ["sale_amount", "discount", "precpt"]:
        neg_count = int((df[c] < 0).sum())
        if neg_count > 0:
            raise ValidationFailure(f"[{name}] {c} has {neg_count} negative values.")
    result["non_negativity_check"] = {
        c: {"negative_count": 0} for c in ["sale_amount", "discount", "precpt"]
    }

    # --- discount anomaly: flag, do not alter -------------------------
    above_ref = df["discount"] > 1.0
    df["discount_above_reference"] = above_ref
    zero_disc = df["discount"] == 0.0
    df["discount_zero"] = zero_disc
    result["discount_anomaly"] = {
        "above_reference_count": int(above_ref.sum()),
        "zero_discount_count": int(zero_disc.sum()),
        "zero_discount_share": round(float(zero_disc.mean()), 6),
        "value_unchanged": True,
        "note": ("Values above the documented 1.0 'no discount' reference are "
                 "retained unchanged; flagged via discount_above_reference. "
                 "Values of exactly 0.0 (a 100% discount) are retained unchanged; "
                 "flagged via discount_zero for treatment at feature stage."),
    }

    # --- row count preservation ------------------------------------
    output_rows = len(df)
    if output_rows != input_rows:
        raise ValidationFailure(
            f"[{name}] Row count changed during cleaning: {input_rows} -> {output_rows}"
        )
    result["output_rows"] = int(output_rows)
    result["output_columns"] = int(df.shape[1])
    result["row_count_preserved"] = True

    # --- date coverage (for the report, informational) -------------------
    result["date_coverage"] = {
        "min": str(df["dt"].min().date()),
        "max": str(df["dt"].max().date()),
        "unique_dates": int(df["dt"].nunique()),
    }

    # --- expected row-count sanity check (informational, non-fatal) ------
    expected = EXPECTED_SPLIT_ROWS.get(name)
    result["expected_row_count_match"] = (expected is None) or (output_rows == expected)

    return df, result


def main():
    ensure_dirs()

    raw_paths = {
        "train": RAW_FRESHRETAILNET_DIR / "data" / "train.parquet",
        "eval": RAW_FRESHRETAILNET_DIR / "data" / "eval.parquet",
    }
    for name, p in raw_paths.items():
        if not p.exists():
            print(f"[STOP] Raw file not found for '{name}': {p}")
            print("Run scripts/01_download_freshretailnet.py first.")
            sys.exit(1)

    cleaned_dfs = {}
    split_reports = {}

    print("Running Phase D cleaning + full-dataset validation (no files written yet)...\n")

    try:
        for name, path in raw_paths.items():
            print(f"Processing '{name}' split...")
            df, report = clean_and_validate_split(name, path)
            cleaned_dfs[name] = df
            split_reports[name] = report
            print(f"  PASS — {report['output_rows']} rows, {report['output_columns']} columns\n")
    except ValidationFailure as e:
        print(f"\n[VALIDATION FAILED] {e}")
        print("No cleaned parquet files were written. Raw data is untouched.")
        # Write a FAIL report so the failure itself is auditable
        fail_report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "overall_status": "FAIL",
            "error": str(e),
            "partial_results": split_reports,
        }
        out_path = VALIDATION_DIR / "freshretailnet_cleaning_validation.json"
        with open(out_path, "w") as f:
            json.dump(fail_report, f, indent=2, default=str)
        print(f"Failure report written to: {out_path}")
        sys.exit(1)

    # --- schema identity check across splits ---------------------------
    train_cols = cleaned_dfs["train"].dtypes.astype(str).to_dict()
    eval_cols = cleaned_dfs["eval"].dtypes.astype(str).to_dict()
    columns_match = list(train_cols.keys()) == list(eval_cols.keys())
    dtypes_match = train_cols == eval_cols

    if not (columns_match and dtypes_match):
        print("[VALIDATION FAILED] Train/eval schema mismatch after cleaning.")
        fail_report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "overall_status": "FAIL",
            "error": "train/eval schema mismatch",
            "train_dtypes": train_cols,
            "eval_dtypes": eval_cols,
        }
        out_path = VALIDATION_DIR / "freshretailnet_cleaning_validation.json"
        with open(out_path, "w") as f:
            json.dump(fail_report, f, indent=2, default=str)
        sys.exit(1)

    # --- ALL VALIDATIONS PASSED: now, and only now, write cleaned files ---
    print("All validations PASSED for both splits. Writing cleaned parquet files...\n")

    train_out = CLEANED_DIR / "freshretailnet_train.parquet"
    eval_out = CLEANED_DIR / "freshretailnet_eval.parquet"
    cleaned_dfs["train"].to_parquet(train_out, index=False)
    cleaned_dfs["eval"].to_parquet(eval_out, index=False)

    final_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train": split_reports["train"],
        "eval": split_reports["eval"],
        "schema_identity_check": {
            "train_eval_columns_match": columns_match,
            "train_eval_dtypes_match": dtypes_match,
        },
        "overall_status": "PASS",
        "output_files": {"train": str(train_out), "eval": str(eval_out)},
    }
    out_path = VALIDATION_DIR / "freshretailnet_cleaning_validation.json"
    with open(out_path, "w") as f:
        json.dump(final_report, f, indent=2, default=str)

    print(f"Cleaned files written:\n  {train_out}\n  {eval_out}")
    print(f"Validation report written to:\n  {out_path}")
    print(f"\nOverall status: PASS")


if __name__ == "__main__":
    main()