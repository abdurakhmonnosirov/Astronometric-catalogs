from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from pathlib import Path
from astropy.table import Table, vstack
from astroquery.xmatch import XMatch
import pandas as pd
import time
import math

PM_LIMIT = 500  # mas/year
W1_LIMIT = 17
output_dir = Path(f"w1{W1_LIMIT:.0f}_pm{PM_LIMIT:.0f}_Q-4")
attempts = 10

# Choose a chunk size that is comfortably below the CDS/XMatch request size limit.
# You may need to tune this depending on number of columns / string sizes.
CHUNK_SIZE = 400000

Output_names = pd.read_csv(output_dir / "Output_names.csv")
matched_tables = []

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

    coord_2018 = coord.apply_space_motion(new_obstime=Time("J2018", format="jyear_str"))
    astropy_filtered["ra_J2018"] = coord_2018.ra.deg
    astropy_filtered["dec_J2018"] = coord_2018.dec.deg
    print(f"Coordinates for {File} propagated to J2018.")

    file_matched_chunks = []
    n_chunks = math.ceil(len(astropy_filtered) / CHUNK_SIZE)

    for i, (start, end, chunk) in enumerate(split_table(astropy_filtered, CHUNK_SIZE), start=1):
        print(f"Cross-matching chunk {i}/{n_chunks} for {File} (rows {start}:{end-1})")

        try:
            matched_chunk = safe_xmatch(
                cat1=chunk,
                cat2="vizier:J/ApJS/255/30/comp",
                max_distance=2 * u.arcsec,
                colRA1="ra_J2018",
                colDec1="dec_J2018",
                colRA2="RAJ2000",
                colDec2="DEJ2000",
            )

            print(f"  -> matched rows in chunk: {len(matched_chunk)}")

            if len(matched_chunk) > 0:
                file_matched_chunks.append(matched_chunk)

        except Exception as e:
            print(f"Chunk {i} failed for {File}: {e}")

    if file_matched_chunks:
        file_combined = vstack(file_matched_chunks)
        outname = output_dir / f"{Path(File).stem}_VLASS_match.csv"
        file_combined.write(outname, format="ascii.csv", overwrite=True)
        matched_tables.append(file_combined)
        print(f"Saved combined matches for {File} -> {outname}")
    else:
        print(f"No matches found for {File}")

# Merge all files together
if matched_tables:
    combined_table = vstack(matched_tables)
    combined_table.write(output_dir / "CatWISE_VLASS_2arcsec.csv", format="ascii.csv", overwrite=True)
    print(f"\nFinal combined catalog saved to {output_dir / 'CatWISE_VLASS_2arcsec.csv'}")
else:
    print("No matches found in any file.")
