"""
Phase C — Raw data inspection: FreshRetailNet-50K

Read-only inspection of the raw parquet files. Produces a structured
JSON report and a console summary. Nothing is modified, cleaned, or
written back to data/raw/.

Run:
    python scripts/03_inspect_freshretailnet.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config.paths import RAW_FRESHRETAILNET_DIR, INSPECTION_DIR, ensure_dirs

KEY_COLUMNS = ["city_id", "store_id", "product_id", "dt"]
ARRAY_COLUMNS = ["hours_sale", "hours_stock_status"]
FLAG_COLUMNS = ["holiday_flag", "activity_flag"]
NUMERIC_RANGE_COLUMNS = [
    "sale_amount", "discount", "stock_hour6_22_cnt",
    "precpt", "avg_temperature", "avg_humidity", "avg_wind_level",
]


def inspect_split(name, path):
    df = pd.read_parquet(path)
    report = {"split": name, "path": str(path)}

    # --- shape & schema -----------------------------------------
    report["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    report["columns"] = df.columns.tolist()
    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

    # --- missing values -------------------------------------------
    report["missing_values"] = {c: int(df[c].isna().sum()) for c in df.columns}

    # --- date field ---------------------------------------------
    parsed_dates = pd.to_datetime(df["dt"], errors="coerce")
    n_unparseable = int(parsed_dates.isna().sum())
    report["date_field"] = {
        "raw_dtype": str(df["dt"].dtype),
        "unparseable_count": n_unparseable,
        "min_date": str(parsed_dates.min()) if n_unparseable < len(df) else None,
        "max_date": str(parsed_dates.max()) if n_unparseable < len(df) else None,
        "n_unique_dates": int(parsed_dates.nunique()),
    }

    # --- composite key uniqueness (the real duplicate test) ------
    key_dupes = df.duplicated(subset=KEY_COLUMNS).sum()
    report["composite_key_check"] = {
        "key_columns": KEY_COLUMNS,
        "duplicate_key_rows": int(key_dupes),
    }

    # --- cardinality of identifier columns ------------------------
    report["cardinality"] = {
        c: int(df[c].nunique())
        for c in ["city_id", "store_id", "product_id",
                  "management_group_id", "first_category_id",
                  "second_category_id", "third_category_id"]
    }

    # --- flag columns: confirm binary ------------------------------
    report["flag_columns"] = {
        c: sorted(df[c].dropna().unique().tolist()) for c in FLAG_COLUMNS
    }

    # --- numeric range checks --------------------------------------
    range_report = {}
    for c in NUMERIC_RANGE_COLUMNS:
        s = df[c]
        range_report[c] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "negative_count": int((s < 0).sum()),
            "zero_count": int((s == 0).sum()),
        }
    report["numeric_ranges"] = range_report

    # 1.0 is documented by the FreshRetailNet README as "no discount"
    # (0.9 = 10% off). The README does NOT define an explicit upper bound,
    # so values above 1.0 are reported as unusual relative to the documented
    # reference point — not asserted as invalid.
    below_zero = int((df["discount"] < 0).sum())
    above_reference = int((df["discount"] > 1.0).sum())
    report["numeric_ranges"]["discount"]["below_zero_count"] = below_zero
    report["numeric_ranges"]["discount"]["above_documented_reference_1_0_count"] = above_reference
    
    # --- discount-above-reference anomaly: structured evidence, no treatment ---
    above_ref_mask = df["discount"] > 1.0
    if above_ref_mask.any():
        sub = df.loc[above_ref_mask]
        dates_parsed = pd.to_datetime(sub["dt"], errors="coerce")
        report["discount_above_reference_anomaly"] = {
            "row_count": int(above_ref_mask.sum()),
            "value_counts": sub["discount"].round(3).value_counts().sort_index().to_dict(),
            "by_product_id": sub["product_id"].value_counts().head(10).to_dict(),
            "by_city_id": sub["city_id"].value_counts().head(10).to_dict(),
            "date_min": str(dates_parsed.min()),
            "date_max": str(dates_parsed.max()),
            "n_unique_dates": int(dates_parsed.nunique()),
            "note": ("Values above the documented 1.0 'no discount' reference. "
                     "Treatment (retain / flag / reinterpret as a price ratio) "
                     "deferred to Phase D; raw data left unmodified."),
        }

    # --- array column structural checks -----------------------------
    array_report = {}
    for c in ARRAY_COLUMNS:
        sample = df[c].iloc[:5000]  # bounded sample for speed on 4.5M rows
        lengths = sample.map(len)
        length_counts = lengths.value_counts().to_dict()
        array_report[c] = {
            "sample_size": len(sample),
            "length_value_counts": {int(k): int(v) for k, v in length_counts.items()},
            "expected_length_24_consistent": bool((lengths == 24).all()),
        }
        if c == "hours_stock_status":
            distinct_vals = set()
            for arr in sample:
                distinct_vals.update(np.unique(arr).tolist())
            array_report[c]["observed_values"] = sorted(float(v) for v in distinct_vals)
    report["array_columns"] = array_report

    # --- cross-field consistency: stock_hour6_22_cnt vs hours_stock_status[6:22] ---
    sample_idx = df.index[:5000]
    sub = df.loc[sample_idx]
    computed = sub["hours_stock_status"].map(lambda a: int(np.sum(a[6:22])))
    mismatch = int((computed != sub["stock_hour6_22_cnt"]).sum())
    report["cross_field_consistency"] = {
        "check": "stock_hour6_22_cnt == sum(hours_stock_status[6:22])",
        "sample_size": len(sample_idx),
        "mismatch_count": mismatch,
    }

    return report, df


def main():
    ensure_dirs()

    train_path = RAW_FRESHRETAILNET_DIR / "data" / "train.parquet"
    eval_path = RAW_FRESHRETAILNET_DIR / "data" / "eval.parquet"

    if not train_path.exists() or not eval_path.exists():
        print(f"[STOP] Expected raw files not found at:\n  {train_path}\n  {eval_path}")
        print("Run scripts/01_download_freshretailnet.py first.")
        sys.exit(1)

    print("Inspecting FreshRetailNet-50K raw files (read-only)...\n")
    train_report, train_df = inspect_split("train", train_path)
    eval_report, eval_df = inspect_split("eval", eval_path)

    # --- train/eval key overlap check ---------------------------------
    train_keys = set(map(tuple, train_df[KEY_COLUMNS].values.tolist()))
    eval_keys = set(map(tuple, eval_df[KEY_COLUMNS].values.tolist()))
    overlap = train_keys & eval_keys

    combined_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train": train_report,
        "eval": eval_report,
        "train_eval_key_overlap": {
            "key_columns": KEY_COLUMNS,
            "overlapping_key_count": len(overlap),
        },
    }

    out_path = INSPECTION_DIR / "freshretailnet_inspection_report.json"
    with open(out_path, "w") as f:
        json.dump(combined_report, f, indent=2)

    # --- console summary ------------------------------------------------
    for name, rep in [("TRAIN", train_report), ("EVAL", eval_report)]:
        print(f"=== {name} ===")
        print(f"  Shape: {rep['shape']}")
        print(f"  Date range: {rep['date_field']['min_date']} to {rep['date_field']['max_date']}"
              f" ({rep['date_field']['n_unique_dates']} unique dates)")
        print(f"  Duplicate composite-key rows: {rep['composite_key_check']['duplicate_key_rows']}")
        print(f"  Cardinality: {rep['cardinality']}")
        print(f"  Flag columns: {rep['flag_columns']}")
        print(f"  Discount above documented reference 1.0: "
              f"{rep['numeric_ranges']['discount']['above_documented_reference_1_0_count']}")
        print(f"  hours_sale/hours_stock_status length==24 consistent: "
              f"{rep['array_columns']['hours_sale']['expected_length_24_consistent']}, "
              f"{rep['array_columns']['hours_stock_status']['expected_length_24_consistent']}")
        print(f"  Cross-field mismatch (stock_hour6_22_cnt vs summed array): "
              f"{rep['cross_field_consistency']['mismatch_count']} / "
              f"{rep['cross_field_consistency']['sample_size']}")
        print()

    print(f"Train/eval composite-key overlap: {len(overlap)} (should be 0 given temporal split)")
    print(f"\nFull report written to:\n  {out_path}")


if __name__ == "__main__":
    main()