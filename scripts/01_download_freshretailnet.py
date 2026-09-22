"""
Phase B — Raw data acquisition: FreshRetailNet-50K

Mirrors the FreshRetailNet-50K HuggingFace dataset repository into
data/raw/freshretailnet/ exactly as distributed. No cleaning, filtering,
reshaping, or interpretation happens here — only download + provenance
recording.

Safe to re-run: huggingface_hub's snapshot_download is incremental
(skips files that are already present and unchanged).

Run:
    python scripts/01_download_freshretailnet.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from huggingface_hub import snapshot_download, HfApi

from config.paths import RAW_FRESHRETAILNET_DIR, METADATA_DIR, ensure_dirs

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "project.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def main():
    ensure_dirs()
    cfg = load_config()
    repo_id = cfg["freshretailnet"]["hf_dataset_id"]

    print(f"Mirroring raw dataset repo '{repo_id}' into:\n  {RAW_FRESHRETAILNET_DIR}")
    print("(This is a raw repo mirror — no transformation is applied. "
          "Safe to re-run; unchanged files are not re-downloaded.)\n")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(RAW_FRESHRETAILNET_DIR),
    )

    api = HfApi()
    try:
        revision = api.dataset_info(repo_id).sha
    except Exception as e:
        revision = f"unavailable ({e})"

    files = sorted(
        str(p.relative_to(RAW_FRESHRETAILNET_DIR))
        for p in RAW_FRESHRETAILNET_DIR.rglob("*")
        if p.is_file()
    )

    manifest = {
        "dataset": repo_id,
        "source_method": "huggingface_hub.snapshot_download",
        "revision_sha": revision,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_path": str(RAW_FRESHRETAILNET_DIR),
        "file_count": len(files),
        "files": files,
    }

    manifest_path = METADATA_DIR / "freshretailnet_raw_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. {len(files)} raw files present at:\n  {RAW_FRESHRETAILNET_DIR}")
    print(f"Manifest written to:\n  {manifest_path}")


if __name__ == "__main__":
    main()