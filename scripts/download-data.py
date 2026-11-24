import os

import requests


BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/suppl"
FILENAMES = [
    "GSE131907_Lung_Cancer_raw_UMI_matrix.rds.gz",
    "GSE131907_Lung_Cancer_cell_annotation.txt.gz",
    "GSE131907_Lung_Cancer_Feature_Summary.xlsx",
]
RAW_DIR = "data/raw"


def download_file(filename):
    destination = os.path.join(RAW_DIR, filename)
    # Keep existing raw files so rerunning the script does not redownload them.
    if os.path.exists(destination):
        print(f"Already exists: {destination}")
        return

    temporary_file = f"{destination}.part"
    response = requests.get(f"{BASE_URL}/{filename}", stream=True, timeout=60)
    response.raise_for_status()

    # Stream each archive to disk instead of loading it entirely into memory.
    with open(temporary_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            file.write(chunk)

    # Expose the final filename only after the download completes.
    os.replace(temporary_file, destination)
    print(f"Downloaded {destination}")
    

if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    for filename in FILENAMES:
        download_file(filename)
