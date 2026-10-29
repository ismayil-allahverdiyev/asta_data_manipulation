from pathlib import Path
import pandas as pd


# Filtering metadata to seperate into easy and hard negatives
def filter_metadata(keyword: str, read_directory: str, write_directory: str) -> None:
    metadata_path = Path(read_directory)
    metadata_pd = pd.read_csv(metadata_path, sep="\t")
    
    metadata_pd["type"] = 0 # 0 symbolizes the easy negative(it is not at all similar to asta and shouldn't activate)
    total = 0
    counter = 0
    for idx, sentence in metadata_pd["sentence"].items():
        total+=1
        if(sentence.find(keyword) != -1):
            counter+=1
            metadata_pd.loc[idx, "type"] = 1 # 1 symbolizes the hard negative(it is similar to asta but shouldn't activate)
    
    print(str(counter) + " hard negatives out of " + str(total))
    
    metadata_pd.to_csv(
        write_directory,
        sep="\t",
        index=False,
    )