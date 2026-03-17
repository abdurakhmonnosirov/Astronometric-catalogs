from pathlib import Path
import numpy as np
from tqdm import tqdm
import gzip

def count_ipac_rows(file_path):
    in_data = False
    n_rows = 0

    with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()

            if not s:
                continue

            # IPAC header/comment lines
            if not in_data:
                if s.startswith("\\") or s.startswith("|"):
                    continue
                in_data = True

            n_rows += 1

    return n_rows

Folders = np.arange(0, 359)
sources_cat = 0

for Folder in tqdm(Folders, desc="Folders", unit="folder"):
    path = Path(f"/Volumes/PortableSSD/CatWISE/2020/{Folder:03.0f}")
    catwise = sorted(f for f in path.iterdir() if f.is_file() and "_cat_" in f.name)
    sources_folder = 0

    for File in tqdm(catwise, desc=f"{Folder:03.0f}", unit="file", leave=False):
        n_rows = count_ipac_rows(File)
        sources_cat += n_rows
        sources_folder += n_rows

    print(f"{Folder:03.0f} has {sources_folder} sources.")

print(f"Total number of sources in CatWISE2020 catalog: {sources_cat}")