"""
Phase F — GDELT daily China news features

Turns the cleaned GDELT GKG records into ONE ROW PER CHINA CALENDAR DAY of
news context features, ready to be joined to FreshRetailNet by date.

Steps:
  1. Read only the 5 needed columns from data/cleaned/gdelt_cleaned.parquet,
     in batches (the file is about 3.7 GB, so it is never loaded whole).
  2. Convert each record's UTC timestamp to China time (Asia/Shanghai) and
     take the China calendar date.
  3. Keep records whose V2Locations mention China (country code "CH" in the
     third '#' field of any location block). Hong Kong (HK) and Taiwan (TW)
     have their own codes in GDELT and are not included.
  4. Remove duplicate articles (same DocumentIdentifier) as a safety check.
  5. Classify each China article into four theme groups by exact word
     matching on its V2Themes names: food, weather, prices, supply.
  6. Aggregate per China date into shares and averages. Raw counts are kept
     only for quality checks, because the corpus is a systematic sample
     (4 of the 96 daily GKG files).

Leakage note:
  This script does NOT shift dates. The one-day lag (yesterday's news for
  today's sales) is applied when the news is joined to the retail data in
  script 10. No statistic is computed across days here.

Outputs:
  data/processed/gdelt_daily_china_features.parquet
  data/processed/gdelt_daily_china_features.csv     (same table, easy to open)
  results/validation/gdelt_daily_features_report.json

Run:
  python scripts/09_build_gdelt_daily_features.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from config.paths import CLEANED_DIR, PROCESSED_DIR, VALIDATION_DIR, ensure_dirs

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "project.yaml"
INPUT_PATH = CLEANED_DIR / "gdelt_cleaned.parquet"
OUT_PARQUET = PROCESSED_DIR / "gdelt_daily_china_features.parquet"
OUT_CSV = PROCESSED_DIR / "gdelt_daily_china_features.csv"
REPORT_PATH = VALIDATION_DIR / "gdelt_daily_features_report.json"

COLUMNS = ["gkg_datetime_utc", "document_identifier", "v2_themes_raw",
           "v2_locations_raw", "tone_average"]
BATCH_SIZE = 50_000
CHINA_CODE = "CH"
SNAPSHOTS_PER_DAY = 4
MIN_CHINA_ARTICLES_PER_DAY = 50

# Theme groups: a GKG theme name (for example "NATURAL_DISASTER_FLOOD") is
# split on "_" and the article is counted in a group when any word of any
# of its themes is in the group's word list, or a theme starts with one of
# the group's prefixes. Exact words avoid false hits such as "RAIN" inside
# "UKRAINE". A few themes that contain a group word but mean something else
# are excluded (checked on a sample of China articles).
THEME_GROUPS = {
    "food": {"FOOD", "AGRICULTURE", "AGRICULTURAL", "FARM", "FARMER", "FARMERS",
             "FARMING", "CROP", "CROPS", "HARVEST", "LIVESTOCK", "FISHERIES",
             "GRAIN", "RICE", "VEGETABLES", "MEAT", "PORK"},
    "weather": {"WEATHER", "FLOOD", "FLOODS", "FLOODING", "DROUGHT", "DROUGHTS",
                "HEATWAVE", "STORM", "STORMS", "TYPHOON", "TYPHOONS", "CYCLONE",
                "HURRICANE", "RAINFALL", "SNOW", "LANDSLIDE"},
    "prices": {"INFLATION", "PRICE", "PRICES", "PRICING", "COST", "COSTS"},
    "supply": {"SUPPLY", "SUPPLIES", "SHORTAGE", "SHORTAGES", "LOGISTICS",
               "SHIPPING", "TRANSPORT", "TRANSPORTATION", "FREIGHT"},
}
THEME_PREFIXES = {"weather": ("NATURAL_DISASTER",)}
THEME_EXCLUDE = {
    "prices": {"ECON_HOUSING_PRICES", "WB_1846_CARBON_PRICING"},
    "supply": {"PUBLIC_TRANSPORT"},
}


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------

def mentions_china(v2_locations: str) -> bool:
    """True if any location block has country code CH (3rd '#' field)."""
    if not v2_locations:
        return False
    for block in v2_locations.split(";"):
        parts = block.split("#")
        if len(parts) >= 3 and parts[2] == CHINA_CODE:
            return True
    return False


def theme_names(v2_themes: str) -> set:
    """Theme names of one article. V2Themes = 'THEME,offset;THEME,offset'."""
    names = set()
    if not v2_themes:
        return names
    for entry in v2_themes.split(";"):
        if entry:
            names.add((entry.rpartition(",")[0] or entry).upper())
    return names


def in_group(names: set, group: str) -> bool:
    """True if any theme name belongs to the theme group."""
    vocab = THEME_GROUPS[group]
    prefixes = THEME_PREFIXES.get(group, ())
    exclude = THEME_EXCLUDE.get(group, set())
    for name in names:
        if name in exclude:
            continue
        if prefixes and name.startswith(prefixes):
            return True
        if not vocab.isdisjoint(name.split("_")):
            return True
    return False


def iter_batches(path):
    """Yield pandas DataFrames of the needed columns, one batch at a time."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=COLUMNS):
        yield batch.to_pandas()


# ---------------------------------------------------------------------
# Core logic (works on any iterable of DataFrames)
# ---------------------------------------------------------------------

def collect(batches, tz_name):
    """Return (daily totals of ALL articles, row-level table of China articles)."""
    total_parts, china_parts = [], []
    rows_read = 0
    for df in batches:
        rows_read += len(df)
        local = pd.to_datetime(df["gkg_datetime_utc"], utc=True).dt.tz_convert(tz_name)
        df = df.assign(china_date=local.dt.tz_localize(None).dt.normalize(),
                       snapshot_utc=pd.to_datetime(df["gkg_datetime_utc"], utc=True))
        total_parts.append(df.groupby("china_date").size())

        is_china = df["v2_locations_raw"].map(mentions_china)
        ch = df.loc[is_china, ["china_date", "snapshot_utc", "document_identifier",
                               "tone_average", "v2_themes_raw"]].copy()
        names = ch.pop("v2_themes_raw").map(theme_names)
        for group in THEME_GROUPS:
            ch[f"is_{group}"] = names.map(lambda n, g=group: in_group(n, g))
        china_parts.append(ch)

    totals = pd.concat(total_parts).groupby(level=0).sum().rename("all_articles")
    china = pd.concat(china_parts, ignore_index=True)
    return totals, china, rows_read


def aggregate(totals, china):
    """One row per China date with shares and averages."""
    snapshots = china.groupby("china_date")["snapshot_utc"].nunique()
    g = china.groupby("china_date")
    daily = pd.DataFrame({
        "news_snapshots": snapshots,
        "news_china_articles": g.size(),
        "news_tone_mean": g["tone_average"].mean(),
        "news_tone_negative_share": g["tone_average"].apply(lambda s: float((s < 0).mean())),
    })
    for group in THEME_GROUPS:
        daily[f"news_{group}_share"] = g[f"is_{group}"].mean()

    # every calendar day in the range appears, even if it had no China news
    full = pd.date_range(totals.index.min(), totals.index.max(), freq="D")
    daily = daily.reindex(full)
    daily["news_all_articles"] = totals.reindex(full).fillna(0).astype(int)
    daily["news_china_articles"] = daily["news_china_articles"].fillna(0).astype(int)
    daily["news_snapshots"] = daily["news_snapshots"].fillna(0).astype(int)
    daily["news_china_share"] = np.where(daily["news_all_articles"] > 0,
                                         daily["news_china_articles"] / daily["news_all_articles"].clip(lower=1),
                                         np.nan)
    daily["complete_day"] = daily["news_snapshots"] == SNAPSHOTS_PER_DAY
    daily.index.name = "dt"
    order = ["news_snapshots", "complete_day", "news_all_articles", "news_china_articles",
             "news_china_share", "news_tone_mean", "news_tone_negative_share"] + \
            [f"news_{g}_share" for g in THEME_GROUPS]
    return daily[order].reset_index()


def validate(daily, needed_start, needed_end):
    """Checks on the days that the retail data will actually use."""
    need = daily[(daily.dt >= needed_start) & (daily.dt <= needed_end)]
    feature_cols = [c for c in daily.columns if c.startswith("news_") and c.endswith(("_share", "_mean"))]
    expected_days = (needed_end - needed_start).days + 1
    checks = {
        "needed_range": [str(needed_start.date()), str(needed_end.date())],
        "needed_days_expected": expected_days,
        "needed_days_present": int(len(need)),
        "needed_days_incomplete_snapshots": need.loc[~need.complete_day, "dt"].dt.strftime("%Y-%m-%d").tolist(),
        "needed_days_below_min_china_articles": need.loc[
            need.news_china_articles < MIN_CHINA_ARTICLES_PER_DAY, "dt"].dt.strftime("%Y-%m-%d").tolist(),
        "missing_feature_values_in_needed_range": int(need[feature_cols].isna().sum().sum()),
        "shares_outside_0_1": int(((need[[c for c in feature_cols if c.endswith("_share")]] < 0) |
                                   (need[[c for c in feature_cols if c.endswith("_share")]] > 1)).sum().sum()),
    }
    hard_ok = (checks["needed_days_present"] == expected_days
               and not checks["needed_days_incomplete_snapshots"]
               and checks["missing_feature_values_in_needed_range"] == 0
               and checks["shares_outside_0_1"] == 0)
    return hard_ok, checks


def main():
    ensure_dirs()
    cfg = load_config()
    tz_name = cfg["gdelt"].get("timezone_for_daily_aggregation", "Asia/Shanghai")

    if not INPUT_PATH.exists():
        print(f"[STOP] {INPUT_PATH} not found. Run scripts/07_clean_gdelt.py first.")
        sys.exit(1)

    print(f"Reading {INPUT_PATH.name} in batches of {BATCH_SIZE:,} (columns: {', '.join(COLUMNS)})...")
    totals, china, rows_read = collect(iter_batches(INPUT_PATH), tz_name)

    dup = int(china["document_identifier"].duplicated().sum())
    china = china.drop_duplicates("document_identifier")
    daily = aggregate(totals, china)

    # FreshRetailNet covers 2024-03-28 to 2024-07-02; with a one-day lag the
    # news days needed are 2024-03-27 to 2024-07-01.
    retail_start = pd.Timestamp(cfg["gdelt"]["start_date"])
    retail_end = pd.Timestamp(cfg["gdelt"]["end_date"])
    needed_start, needed_end = retail_start - pd.Timedelta(days=1), retail_end - pd.Timedelta(days=1)
    ok, checks = validate(daily, needed_start, needed_end)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS" if ok else "FAIL",
        "input": {"path": str(INPUT_PATH), "rows_read": rows_read, "columns_read": COLUMNS},
        "method": {
            "timezone": tz_name,
            "china_rule": "any V2Locations block with country code 'CH' (HK and TW excluded)",
            "theme_rule": "exact word match on V2Themes theme names split by '_', plus prefixes, minus exclusions",
            "theme_groups": {k: sorted(v) for k, v in THEME_GROUPS.items()},
            "theme_prefixes": {k: list(v) for k, v in THEME_PREFIXES.items()},
            "theme_exclusions": {k: sorted(v) for k, v in THEME_EXCLUDE.items()},
            "lag_applied_here": False,
        },
        "china_articles": {
            "total_after_dedup": int(len(china)),
            "duplicates_removed": dup,
            "share_of_all_records": round(len(china) / rows_read, 4),
            "per_day_mean": round(float(daily.news_china_articles.mean()), 1),
            "per_day_min_in_needed_range": int(daily.loc[(daily.dt >= needed_start) & (daily.dt <= needed_end),
                                                         "news_china_articles"].min()),
        },
        "output": {"days": int(len(daily)), "first_day": str(daily.dt.min().date()),
                   "last_day": str(daily.dt.max().date()), "columns": list(daily.columns),
                   "parquet": str(OUT_PARQUET), "csv": str(OUT_CSV)},
        "feature_means_in_needed_range": {
            c: round(float(daily.loc[(daily.dt >= needed_start) & (daily.dt <= needed_end), c].mean()), 4)
            for c in daily.columns if c.endswith(("_share", "_mean"))},
        "checks": checks,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if not ok:
        print("\n[VALIDATION FAILED] Daily features not written. See:", REPORT_PATH)
        print(json.dumps(checks, indent=2))
        sys.exit(1)

    daily.to_parquet(OUT_PARQUET, index=False)
    daily.to_csv(OUT_CSV, index=False)

    print("\n=== GDELT DAILY CHINA FEATURES ===")
    print(f"Records read: {rows_read:,}")
    print(f"China articles kept: {len(china):,} ({report['china_articles']['share_of_all_records']:.1%} of records), "
          f"duplicates removed: {dup}")
    print(f"Days: {len(daily)} ({report['output']['first_day']} to {report['output']['last_day']}, China time)")
    print(f"Days needed for the retail join: {checks['needed_days_present']}/{checks['needed_days_expected']} "
          f"({checks['needed_range'][0]} to {checks['needed_range'][1]}), all with 4 snapshots")
    print(f"China articles per day: mean {report['china_articles']['per_day_mean']}, "
          f"min {report['china_articles']['per_day_min_in_needed_range']}")
    if checks["needed_days_below_min_china_articles"]:
        print(f"Note: days with fewer than {MIN_CHINA_ARTICLES_PER_DAY} China articles: "
              f"{checks['needed_days_below_min_china_articles']}")
    print("\nAverage feature values (needed range):")
    for k, v in report["feature_means_in_needed_range"].items():
        print(f"  {k:28s} {v}")
    print("\nFirst 5 days:")
    print(daily.head().to_string(index=False))
    print(f"\nSaved:\n  {OUT_PARQUET}\n  {OUT_CSV}\n  {REPORT_PATH}")
    print("\nOverall status: PASS")


if __name__ == "__main__":
    main()