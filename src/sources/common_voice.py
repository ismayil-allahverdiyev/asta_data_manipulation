
import os

from pathlib import Path
import tarfile
import pandas as pd
import requests
from dotenv import load_dotenv
from dataset.sampling import identifying_dataset, sampling_dataset
from filtering.wake_word import filter_metadata


load_dotenv()


BASE_URL = "https://mozilladatacollective.com/api"
DATASET_ID = "cmu5jplf300nwmh07iqvk9leo"
DATASET_DIR = "C:/Users/Isi/Downloads/1789502799671-cv-corpus-27.0-2026-09-11-en.tar.gz"
METADATA_DIR = "./data/raw/common_voice/validated.tsv"
COMMON_VOICE_LABELED_METADATA_PATH = (
    "./data/processed/common_voice/common_voice_labeled.tsv"
)

HARD_NEGATIVE_DIR = "./data/processed/common_voice/hard_negatives"
EASY_NEGATIVE_DIR = "./data/processed/common_voice/easy_negatives"

def get_dataset_info() -> dict:
    url = f"{BASE_URL}/datasets/{DATASET_ID}"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    return response.json()

# Run this URL to request a link to install the whole mozill common voice archive
def request_download_url() -> dict:
    """Request an authenticated, temporary download URL."""
    api_key = os.getenv("MDC_API_KEY")

    if not api_key:
        raise RuntimeError(
            "MDC_API_KEY is not set. Configure your API key first."
        )

    url = f"{BASE_URL}/datasets/{DATASET_ID}/download"

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    response.raise_for_status()

    return response.json()

# Get initial overview of its contents
def inspect_archive(archive_path: str | Path, limit: int = 30) -> None:
    path = Path(archive_path)

    if not path.is_file():
        raise FileNotFoundError(f"Archive not found: {path}")

    size_gb = path.stat().st_size / (1024 ** 3)

    print(f"Archive: {path.name}")
    print(f"Size: {size_gb:.2f} GiB")
    print(f"Location: {path.resolve()}")
    print("\nFirst archive entries:")

    with tarfile.open(path, mode="r|*") as archive:
        for index, member in enumerate(archive):
            if index >= limit:
                break

            entry_type = "directory" if member.isdir() else "file"
            print(
                f"{entry_type:9} "
                f"{member.size:12,} bytes  "
                f"{member.name}"
            )
            
# Loading a TSV metadata file directly from a Common Voice tar archive.
def load_validated_metadata(
    archive_path: str | Path,
    filename: str = "validated.tsv",
) -> pd.DataFrame:
    archive_path = Path(archive_path)
    output_dir = Path("./data/raw/common_voice")

    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    with tarfile.open(archive_path, mode="r|*") as archive:
        for member in archive:
            if Path(member.name).name == filename:
                extracted_file = archive.extractfile(member)
                if extracted_file is None:
                    raise RuntimeError(f"Could not read {filename}")

                with extracted_file, output_path.open("wb") as destination:
                    while chunk := extracted_file.read(1024 * 1024):
                        destination.write(chunk)

                break
        else:
            raise FileNotFoundError(
                f"{filename} not found inside {archive_path.name}"
            )

    return pd.read_csv(output_path, sep="\t")

# Filtering metadata to seperate into easy and hard negatives
def filter_common_voice_metadata() -> None:
    filter_metadata(
        keyword="asta",
        read_directory=METADATA_DIR,
        write_directory=COMMON_VOICE_LABELED_METADATA_PATH
    )

# Splitting the filtered metadata into hard and easy negatives, and extracting the corresponding MP3 files from the Common Voice archive.
def split_mp3_files() -> None:
    res = identifying_dataset(
        dataset=pd.read_csv(
            COMMON_VOICE_LABELED_METADATA_PATH, sep="\t"
        ),
        sample_size=100000,
        random_state=42,
    )
    
    print(f"Hard negatives: {len(res['hard_negatives']):,}")
    print(f"Easy negatives: {len(res['easy_negatives']):,}")
    
    sampling_dataset(
        hard_negatives_dataset=res["hard_negatives"],
        easy_negatives_dataset=res["easy_negatives"],
        hard_negative_dir=HARD_NEGATIVE_DIR,
        easy_negative_dir=EASY_NEGATIVE_DIR,
        data_dir=DATASET_DIR
    )


if __name__ == "__main__":
    filter_common_voice_metadata()