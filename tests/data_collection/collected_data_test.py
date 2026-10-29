from pathlib import Path
import pandas as pd

def test_common_voice_data() -> None:
    metadata_path = Path(
        "data/processed/common_voice/common_voice_labeled.tsv"
    )

    df = pd.read_csv(metadata_path, sep="\t", low_memory=False)

    print("Total recordings:", len(df))
    print("\nLabel distribution:")
    print(df["type"].value_counts(dropna=False))

    print("\nAvailable columns:")
    print(df.columns.tolist())

    print("\nExamples of hard negatives:")
    print(df.loc[df["type"] == 1, "sentence"].head(20).to_string(index=False))
    
if __name__ == '__main__':
    test_common_voice_data()