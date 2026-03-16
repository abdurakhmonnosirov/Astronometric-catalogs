from pathlib import Path
import numpy as np
import pandas as pd
from read_files import read_ipac_columns

PM_LIMIT = 500.0   # mas/year
W1_LIMIT = 17.0
GROUP = 1

folders = np.arange(0, 70)
output_dir = Path(f"w1{W1_LIMIT:.0f}_pm{PM_LIMIT:.0f}_Q-4")
output_dir.mkdir(parents=True, exist_ok=True)

pm_limit_sq = (PM_LIMIT / 1000.0) ** 2  # convert mas/yr threshold to (arcsec/yr)^2 if PMRA/PMDec are arcsec/yr

output_names = []
batch_dfs = []
batch_start = None
batch_count = 0

for folder in folders:
    path = Path(f"/Volumes/PortableSSD/CatWISE/2020/{folder:03d}")

    if not path.exists():
        print(f"Skipping missing folder: {path}")
        continue

    catwise_files = sorted(f for f in path.iterdir() if f.is_file() and "_cat_" in f.name)

    if batch_start is None:
        batch_start = folder

    for file in catwise_files:
        
        needed_cols = ['source_name', 'source_id', 'ra_pm', 'dec_pm', "PMRA", "PMDec", "sigPMRA", "sigPMDec", "w1mpro_pm", 'w1sigmpro_pm', 'w2mpro_pm', 'w2sigmpro_pm', 'rchi2_pm', 'rchi2', 'w1rchi2_pm', 'w2rchi2_pm']
        tbl = read_ipac_columns(file, needed_cols)

        pmra = np.asarray(tbl["PMRA"])
        pmdec = np.asarray(tbl["PMDec"])
        sig_pmra = np.asarray(tbl["sigPMRA"])
        sig_pmdec = np.asarray(tbl["sigPMDec"])
        w1 = np.asarray(tbl["w1mpro_pm"])

        # Avoid divide-by-zero warnings / inf where sig == 0
        good_err = (sig_pmra > 0) & (sig_pmdec > 0)

        pm_sq = pmra**2 + pmdec**2
        # chi_pm_sq = np.full(len(tbl), np.inf, dtype=np.float64)
        # chi_pm_sq[good_err] = (pmra[good_err] / sig_pmra[good_err])**2 + (pmdec[good_err] / sig_pmdec[good_err])**2
        chi_pm_sq = (pmra / sig_pmra)**2 + (pmdec / sig_pmdec)**2


        # Q < 1e-4  <=>  chi_pm_sq > -2 * ln(1e-4)
        mask = (
            ~(chi_pm_sq <= -2.0 * np.log(1e-4))
            & ~(pm_sq < pm_limit_sq)
            & ~(w1 < W1_LIMIT)
        )

        if np.any(mask):
            filtered = tbl[mask]

            # Only compute pm_tot_masyr for the rows you keep
            filtered["pm_tot_masyr"] = np.sqrt(
                np.asarray(filtered["PMRA"])**2 + np.asarray(filtered["PMDec"])**2
            ) * 1000.0

            batch_dfs.append(filtered.to_pandas())

    batch_count += 1

    if batch_count % GROUP == 0:
        if batch_dfs:
            merged = pd.concat(batch_dfs, ignore_index=True)
        else:
            merged = pd.DataFrame()

        output_file = output_dir / f"catwise_{batch_start:03d}_{folder:03d}_filtered.csv"
        merged.to_csv(output_file, index=False)
        output_names.append(output_file.name)

        print(f"Saved {output_file.name} with {len(merged)} rows.")

        batch_dfs.clear()
        batch_start = None

# Flush remainder
if batch_start is not None:
    if batch_dfs:
        merged = pd.concat(batch_dfs, ignore_index=True)
    else:
        merged = pd.DataFrame()

    output_file = output_dir / f"catwise_{batch_start:03d}_{folders[-1]:03d}_filtered.csv"
    merged.to_csv(output_file, index=False)
    output_names.append(output_file.name)

    print(f"Saved {output_file.name} with {len(merged)} rows.")

pd.DataFrame({"File Name": output_names}).to_csv(output_dir / "Output_names.csv", index=False)
