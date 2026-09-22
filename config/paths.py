"""
Centralised, relative path configuration for the project.

Every script must import paths from here rather than hard-coding
absolute paths. This file derives everything from its own location,
so the project can be cloned to any machine and just work.
"""

from pathlib import Path

# Project root = two levels up from this file (config/paths.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Data layers -------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
RAW_FRESHRETAILNET_DIR = RAW_DIR / "freshretailnet"
RAW_GDELT_DIR = RAW_DIR / "gdelt"

CLEANED_DIR = DATA_DIR / "cleaned"
PROCESSED_DIR = DATA_DIR / "processed"

# --- Outputs -------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"
METADATA_DIR = RESULTS_DIR / "metadata"
VALIDATION_DIR = RESULTS_DIR / "validation"

# --- Ensure directories exist (idempotent, safe to call repeatedly) ---
def ensure_dirs():
    for d in [
        RAW_FRESHRETAILNET_DIR,
        RAW_GDELT_DIR,
        CLEANED_DIR,
        PROCESSED_DIR,
        METADATA_DIR,
        VALIDATION_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"PROJECT_ROOT        = {PROJECT_ROOT}")
    print(f"RAW_FRESHRETAILNET   = {RAW_FRESHRETAILNET_DIR}")
    print(f"RAW_GDELT            = {RAW_GDELT_DIR}")
    print(f"CLEANED_DIR          = {CLEANED_DIR}")
    print(f"PROCESSED_DIR        = {PROCESSED_DIR}")
    print(f"METADATA_DIR         = {METADATA_DIR}")
    print(f"VALIDATION_DIR       = {VALIDATION_DIR}")