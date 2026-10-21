# Retail Agentic Replenishment — Data Foundation

MSc Business Analytics Project (BUSI1783), University of Greenwich.

**Project:** Beyond Recorded Sales: Correcting Censored Demand for Machine Learning
Forecasting and Stockout Prediction in Fresh Food Retail.

## Current scope
This repository currently contains the **data foundation only**: raw import,
inspection, validation and cleaning of

- **FreshRetailNet-50K** (primary operational retail dataset, Hugging Face
  `Dingdong-Inc/FreshRetailNet-50K`, CC BY 4.0), and
- **GDELT 2.0 Global Knowledge Graph** (supporting external news context,
  joined to the retail data by date only).

Forecasting, censored demand correction, modelling and the dashboard are
added in later phases.

## Pipeline layers
`data/raw/` → `data/cleaned/` → `data/processed/`

Raw data is never modified. Every layer is produced by a dedicated script,
and every script writes a JSON report to `results/` so each step is auditable.
Cleaning scripts validate the full dataset first and write nothing if any
hard check fails.

## Setup (Windows, VS Code)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python config/paths.py          # creates the folder structure
```

## Run order
| Step | Script | Output |
|---|---|---|
| 1 | `scripts/01_download_freshretailnet.py` | `data/raw/freshretailnet/`, `results/metadata/freshretailnet_raw_manifest.json` |
| 2 | `scripts/02_download_gdelt.py` | `data/raw/gdelt/` (412 files), `results/metadata/gdelt_raw_manifest.json` |
| 3 | `scripts/03_inspect_freshretailnet.py` | `results/inspection/freshretailnet_inspection_report.json` |
| 4 | `scripts/04_inspect_gdelt.py` | `results/inspection/gdelt_inspection_report.json` |
| 5 | `scripts/05_clean_freshretailnet.py` | `data/cleaned/freshretailnet_{train,eval}.parquet`, `results/validation/freshretailnet_cleaning_validation.json` |
| 6 | `scripts/06_validate_gdelt_corpus.py` | `results/inspection/gdelt_full_corpus_validation.json` |
| 6b | `scripts/06b_diagnose_v2locations.py` | `results/inspection/gdelt_v2locations_failure_modes_report.json` |
| 6c | `scripts/06c_diagnose_v2locations_row_impact.py` | `results/inspection/gdelt_v2locations_row_impact_report.json` |
| 7 | `scripts/07_clean_gdelt.py` | `data/cleaned/gdelt_cleaned.parquet`, `results/validation/gdelt_cleaning_validation.json` |
| 8 | `scripts/08_show_data_quality.py` | `results/data_quality/data_quality_report.json` |

Data files are not committed (see `.gitignore`); the JSON reports are.

## Key results so far
**FreshRetailNet-50K**
- Train: 4,500,000 store-product-days, 28 Mar to 25 Jun 2024 (90 days).
  Eval: 350,000 rows, 26 Jun to 2 Jul 2024 (7 days). 18 cities, 898 stores, 865 products.
- 0 missing values, 0 duplicate keys, 0 train/eval key overlap, all hourly arrays length 24.
- `stock_hour6_22_cnt` equals the sum of `hours_stock_status[6:22]` on every row
  (1 = out of stock), so the opening window is 06:00 to 22:00 (16 hours).
- 44.3% of train store-product-days have at least one stockout hour.
- Discount anomalies kept unchanged and flagged: 34 rows above 1.0
  (`discount_above_reference`) and rows equal to 0.0 (`discount_zero`).

**GDELT 2.0 GKG**
- 412 raw files: 4 snapshots per day (00, 06, 12, 18 UTC) from 25 Mar to 5 Jul 2024
  (FreshRetailNet period plus a 3-day buffer). This is a systematic sample of 4 of
  the 96 daily files, so later daily features use shares and averages, not counts.
- 606,981 records, all 27 fields present, unique record IDs, valid dates and tone.
- 307 records (0.05%) contain malformed V2Locations blocks; kept and flagged.

## Data leakage rules (apply to all later phases)
1. No statistic is calculated across the train/eval split during cleaning.
2. The eval split is the final untouched test set; it is not used for any fitting or tuning.
3. GDELT timestamps (UTC) are converted to China time (Asia/Shanghai) before daily
   aggregation, and news is joined with a one-day lag.
4. Lag and rolling features use previous days only.
