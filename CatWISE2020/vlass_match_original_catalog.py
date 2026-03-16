from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from pathlib import Path
from astropy.table import Table, vstack
from astroquery.xmatch import XMatch
import pandas as pd
import time


PM_LIMIT = 500 # mas/year
W1_LIMIT = 17
output_dir = Path(f"w1{W1_LIMIT:.0f}_pm{PM_LIMIT:.0f}_Q-4")
attemps = 10
Output_names = pd.read_csv(output_dir/"Output_names.csv")
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

# df = pd.read_csv("/data/home/abdurakhmon/VLASS1_QL/comp.dat", sep='\s+', header=None)
# df.columns = column_names_VLASS
# VLASS1_QL =  Table.from_pandas(df)
# print("VLASS-1 QL table is loaded.")

for File in Output_names['File Name'].values:
    
    astropy_filtered = Table.read(output_dir / File, format="ascii.csv")
    print(f"{File} loaded.")
    coord=SkyCoord(
        ra=astropy_filtered['ra_pm']*u.deg,
        dec=astropy_filtered['dec_pm']*u.deg,
        pm_ra_cosdec=astropy_filtered['PMRA']*u.arcsec/u.yr,
        pm_dec = astropy_filtered['PMDec']*u.arcsec/u.yr,
        frame='icrs',
        obstime=Time(57170, format='mjd')
    )
    
    coord_2018 = coord.apply_space_motion(new_obstime=Time('J2018', format='jyear_str'))
    astropy_filtered['ra_J2018']=coord_2018.ra.deg
    astropy_filtered['dec_J2018']=coord_2018.dec.deg
    print(f"Coordinates of {File} at Epoch 2018 is calcluated and cross-matching is started.")

    def safe_xmatch(**kwargs):
        for attempt in range(attemps):
            try:
                return XMatch.query(**kwargs)
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                time.sleep(10)  # wait 10 seconds before retry
        raise RuntimeError(f"XMatch query failed after {attemps} attempts")

    matched = safe_xmatch(
        cat1=astropy_filtered,
        cat2="vizier:J/ApJS/255/30/comp",
        max_distance=2 * u.arcsec,  
        colRA1='ra_J2018',
        colDec1='dec_J2018',
        colRA2="RAJ2000",          
        colDec2="DEJ2000",
    )


    # matched = XMatch.query(
    #     cat1=astropy_filtered,
    #     # cat2=VLASS1_QL,
    #     cat2= "vizier:J/ApJS/255/30/comp",
    #     max_distance=5 * u.arcsec,  
    #     colRA1='RA_ICRS_J2018',
    #     colDec1='DE_ICRS_J2018',
    #     # colRA2="RAdeg", 
    #     # colDec2="DEdeg", 
    #     colRA2="RAJ2000",          
    #     colDec2="DEJ2000",  
    # )

    print(f"{File} is cross matched. Number of matched rows: {len(matched)}")
    matched_tables.append(matched)
    
if matched_tables:
    combined_table = vstack(matched_tables)
    combined_table.write(output_dir / "CatWISE_VLASS_2arcsec.csv", overwrite=True)
else:
    print("No matches found.")