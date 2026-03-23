from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from pathlib import Path
from astropy.table import Table, vstack, unique
from astroquery.xmatch import XMatch
import pandas as pd
import sys
import time
import math

PM_LIMIT = float(sys.argv[1])  # mas/year
W1_LIMIT = float(sys.argv[2])
catalog = sys.argv[3]

output_dir = Path(f"w1{W1_LIMIT:.0f}_pm{PM_LIMIT:.0f}_Q-4")
attempts = 10

# Choose a chunk size that is comfortably below the CDS/XMatch request size limit.
# You may need to tune this depending on number of columns / string sizes.
CHUNK_SIZE = 400000

Output_names = pd.read_csv(output_dir / "Output_names.csv")
matched_tables = []
saved_chunk_files = []

column_names_VLASS = [
    "recno", "CompName", "CompId", "IslId", "RAdeg", "DEdeg",
    "e_RAdeg", "e_DEdeg", "Ftot", "e_Ftot", "Fpeak",
    "e_Fpeak", "Maj", "e_Maj", "Min", "e_Min",
    "PA", "e_PA", "FtotIsl", "e_FtotIsl", "Islrms",
    "Islmean", "ResIdIslrms", "ResidIslmean", "RAMdeg", "DEMdeg",
    "e_RAMdeg", "e_DEMdeg", "SCode", "Xposn", "e_Xposn",
    "Yposn", "e_Yposn", "XposnMax", "e_XposnMax", "YposnMax",
    "e_YposnMax", "MajImgPlane", "e_MajImgPlane", "MinImgPlane", "e_MinImgPlane",
    "PAImgPlane", "e_PAImgPlane", "DCMaj", "e_DCMaj", "DCMin",
    "e_DCMin", "DCPA", "e_DCPA", "DCMajImgPlane", "e_DCMajImgPlane",
    "DCMinImgPlane", "e_DCMinImgPlane", "DCPAImgPlane", "e_DCPAImgPlane",
    "Tile", "Subtile", "RASdeg", "DESdeg", "NVSSdist",
    "FIRSTdist", "PeakToRing", "DupFlag", "QualFlag", "NNdist",
    "BMaj", "BMin", "BPA", "MainSample", "QLcutout"
]

LoTSS_DR2 = "vizier:J/A+A/659/A1/catalog" # resolution is aruoud 6", sky coverage is ~27% of the northern sky (or 13.7% of the full sky)
VLASS1_QL = "vizier:J/ApJS/255/30/comp"  # best resolution is 2.5", suggested match radius 5", for objects that have dec >= −40° (33,885 deg^2)
RACS_high = "vizier:J/other/PASA/42.38/sourcesh" # resolution is around 11.8"×8.1", sky up to about Dec +48°, "vizier:J/other/PASA/42.38/gcomps" for Gaussian components
RACS_mid = "vizier:J/other/PASA/41.3/sourcesm" # "vizier:J/other/PASA/41.3/gcompsm" for Gaussian components
GaiaDR3 = "vizier:I/355/gaiadr3"
Galex_DR5 = "vizier:II/335/galex_ais" # this is revised version, original is "vizier:II/312/ais"
TwoMASS = "vizier:II/246/out" # catalogue for point sources.
CatWISE2020 = "vizier:II/365/catwise"

if catalog.lower() == 'vlass' :
    match_catalog = VLASS1_QL
    instrument='VLASS'
    ra_match="RAJ2000",
    dec_match="DEJ2000",
    epochtime=2018
elif catalog.lower() == 'gaia':
    match_catalog = GaiaDR3
    instrument='GaiaDR3'
    ra_match = 'RA_ICRS'
    dec_match = 'DE_ICRS'
    epochtime=2016
else:
    print("Choose one of the catalog from the following list: \n gaia, vlass")

def safe_xmatch(**kwargs):
    for attempt in range(attempts):
        try:
            return XMatch.query(**kwargs)
        except Exception as e:
            print(f"Attempt {attempt + 1}/{attempts} failed: {e}")
            time.sleep(30)
    raise RuntimeError(f"XMatch query failed after {attempts} attempts")


def split_table(table, chunk_size):
    """Yield slices of an Astropy table."""
    n = len(table)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield start, end, table[start:end]


for File in Output_names["File Name"].values:
    print(f"\nLoading {File}")
    astropy_filtered = Table.read(output_dir / File, format="ascii.csv")
    print(f"{File} loaded. Rows: {len(astropy_filtered)}")

    # Propagate coordinates to J2018
    coord = SkyCoord(
        ra=astropy_filtered["ra_pm"] * u.deg,
        dec=astropy_filtered["dec_pm"] * u.deg,
        pm_ra_cosdec=astropy_filtered["PMRA"] * u.arcsec / u.yr,
        pm_dec=astropy_filtered["PMDec"] * u.arcsec / u.yr,
        frame="icrs",
        obstime=Time(57170, format="mjd")
    )

    coord_new = coord.apply_space_motion(new_obstime=Time(f"J{epochtime}", format="jyear_str"))
    astropy_filtered[f"ra_J{epochtime}"] = coord_new.ra.deg
    astropy_filtered[f"dec_J{epochtime}"] = coord_new.dec.deg
    print(f"Coordinates for {File} propagated to J{epochtime}.")

    file_matched_chunks = []
    n_chunks = math.ceil(len(astropy_filtered) / CHUNK_SIZE)

    for i, (start, end, chunk) in enumerate(split_table(astropy_filtered, CHUNK_SIZE), start=1):
        print(f"Cross-matching chunk {i}/{n_chunks} for {File} (rows {start}:{end-1})")

        try:
            matched_chunk = safe_xmatch(
                cat1=chunk,
                cat2=match_catalog,
                max_distance=2 * u.arcsec,
                colRA1=f"ra_J{epochtime}",
                colDec1=f"dec_J{epochtime}",
                colRA2=ra_match,
                colDec2=dec_match,
            )
            
            matched_chunk_unique = unique(matched_chunk, keys='source_id', keep='first')
            print(f"---> results for this chunk: {len(matched_chunk)} matches, {len(matched_chunk_unique)} unique matches.")

            if len(matched_chunk) > 0:
                file_matched_chunks.append(matched_chunk)

        except Exception as e:
            print(f"Chunk {i} failed for {File}: {e}")

    if file_matched_chunks:
        file_combined = vstack(file_matched_chunks)
        outname = output_dir / f"{Path(File).stem}_{instrument}_match.csv"
        file_combined.write(outname, format="ascii.csv", overwrite=True)
        saved_chunk_files.append(outname)
        matched_tables.append(file_combined)
        print(f"Saved combined matches for {File} -> {outname}")
    else:
        print(f"No matches found for {File}")

    

# Merge all files together
if matched_tables:
    combined_table = vstack(matched_tables)
    combined_table.write(output_dir / f"CatWISE_{instrument}_2arcsec.csv", format="ascii.csv", overwrite=True)
    print(f"\nFinal combined catalog saved to {output_dir / f'CatWISE_{instrument}_2arcsec.csv'}")
else:
    print("No matches found in any file.")

if matched_tables:
    combined_table = vstack(matched_tables)
    final_path = output_dir / f"CatWISE_{instrument}_2arcsec.csv"
    combined_table.write(final_path, format="ascii.csv", overwrite=True)
    print(f"\nFinal combined catalog saved to {final_path}")

    # Delete intermediate files
    for f in saved_chunk_files:
        try:
            f.unlink()
        except Exception as e:
            print(f"Could not delete {f}: {e}")
else:
    print("No matches found in any file.")

