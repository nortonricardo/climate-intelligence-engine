"""
1.1 - Download raw data files from Google Drive into the local data/ folder.

Files downloaded:
  - station.parquet
  - weather_measurements.parquet
"""

import sys
from pathlib import Path

import gdown

FOLDER_ID = "1KVpWJoyMfUsuT3yLa1zYmJd1Qk33BIMt"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"

FILES = {
    "station.parquet": None,           # file IDs resolved at runtime via folder listing
    "weather_measurements.parquet": None,
}

DATA_DIR = Path(__file__).parent / "data"


def download():
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Downloading files from Google Drive folder: {FOLDER_URL}")
    print(f"Destination: {DATA_DIR.resolve()}\n")

    # gdown lists the folder and downloads all files inside it
    downloaded = gdown.download_folder(
        url=FOLDER_URL,
        output=str(DATA_DIR),
        quiet=False,
        use_cookies=False,
    )

    if not downloaded:
        print("ERROR: No files were downloaded. Check if the folder is publicly accessible.")
        sys.exit(1)

    # Verify expected files are present
    missing = [f for f in FILES if not (DATA_DIR / f).exists()]
    if missing:
        print(f"\nWARNING: Expected files not found after download: {missing}")
    else:
        print("\nAll files downloaded successfully:")
        for name in FILES:
            size_mb = (DATA_DIR / name).stat().st_size / 1_048_576
            print(f"  {name} — {size_mb:.2f} MB")


if __name__ == "__main__":
    download()
