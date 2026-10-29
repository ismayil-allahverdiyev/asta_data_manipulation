import shutil
import tarfile

import pandas as pd
from pathlib import Path

from tqdm import tqdm

# Identifying dataset to a specified size, ensuring that all hard negatives are included in the sample and the rest of the sample is filled with easy negatives.
def identifying_dataset(
    dataset: pd.DataFrame, sample_size: int, random_state: int = 42
) -> dict:
    hard_negatives_dataset = dataset[dataset["type"] == 1]
    
    total_hard_negatives = len(hard_negatives_dataset)
    total_easy_negatives = sample_size - total_hard_negatives
    
    """
    Samples a dataset to a specified size.

    Args:
        dataset (pd.DataFrame): The dataset to sample.
        sample_size (int): The number of samples to select.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing the sampled dataset and counts of hard and easy negatives.
    """
    if sample_size > len(dataset):
        raise ValueError("Sample size cannot be larger than the dataset size.")

    easy_negatives_dataset = dataset[dataset["type"] == 0].sample(n=total_easy_negatives, random_state=random_state)
    
    return {
        "sampled_dataset": pd.concat([hard_negatives_dataset, easy_negatives_dataset]),
        "hard_negatives": hard_negatives_dataset,
        "easy_negatives": easy_negatives_dataset,
    }

def sampling_dataset(
    hard_negatives_dataset: pd.DataFrame, 
    easy_negatives_dataset: pd.DataFrame,
    hard_negative_dir: str,
    easy_negative_dir: str,
    data_dir: str
) -> None:
    easy_negative_dir_path = Path(easy_negative_dir)
    hard_negative_dir_path = Path(hard_negative_dir)
    
    easy_negative_dir_path.mkdir(parents=True, exist_ok=True)
    hard_negative_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Map each desired filename to its destination(s).
    targets: dict[str, set[Path]] = {}

    for dataset, destination_dir in [
        (hard_negatives_dataset, hard_negative_dir_path),
        (easy_negatives_dataset, easy_negative_dir_path),
    ]:
        for filename in dataset["path"].dropna().astype(str):
            name = Path(filename.replace("\\", "/")).name
            destination = destination_dir / name

            if not destination.exists():
                targets.setdefault(name, set()).add(destination)

    total_requested = sum(len(paths) for paths in targets.values())

    print(f"Unique files to extract: {total_requested:,}")

    if not targets:
        print("All requested files already exist.")
        return

    found = set()

    # Stream through the archive once instead of unpacking everything.
    with tarfile.open(data_dir, mode="r|gz") as archive:
        for member in tqdm(archive, desc="Scanning archive"):
            if not member.isfile():
                continue

            filename = Path(member.name).name

            if filename not in targets:
                continue

            source = archive.extractfile(member)

            if source is None:
                continue

            with source:
                content = source.read()

            for destination in targets[filename]:
                with destination.open("wb") as output:
                    output.write(content)

            found.add(filename)

    missing = set(targets) - found
    extracted = sum(len(targets[name]) for name in found)
    missing_count = sum(len(targets[name]) for name in missing)

    print(f"Extracted: {extracted:,}")
    print(f"Missing: {missing_count:,}")