"""
io.py: Filehandling functions

"""

__author__ = "S. Belkner"


from astropy.io import fits
from logdecorator import log_on_end, log_on_error, log_on_start
from logging import DEBUG, ERROR, INFO
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import functools
import os.path
import numpy as np
from typing import Dict, List, Optional, Tuple
from component_separation.cs_util import Planckf, Plancks, Planckr
import healpy as hp
from scipy import stats
import json
import platform

PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKMAPNSIDE = [1024, 2048]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]

uname = platform.uname()
if uname.node == "DESKTOP-KMIGUPV":
    mch = "XPS"
else:
    mch = "NERSC"

with open('config.json', "r") as f:
    cf = json.load(f)


#%% Collect maps
@log_on_start(INFO, "Starting to load data")
@log_on_end(DEBUG, "Data loaded successfully: '{result}' ")
def load_plamap(pa: Dict) -> List[Dict]:
    """Collects planck maps (.fits files) and stores to dictionaries. Mask data must be placed in `PATH/mask/`,
    Map data in `PATH/map/`.
    Args:
        cf (Dict): Configuration as coming from conf.json
        mch (str): ID of the machine the code is executed. Depending on the ID, a different set of configurations is used.
        
    Returns:
        List[Dict]: Planck maps (data and masks) and some header information

    Doctest:
    >>> get_data(cf: Dict, mch: str) 
    NotSureWhatToExpect
    """

    ### Grab all necessary parameters from config

    mskset = pa['mskset'] # smica or lens
    freqdset = pa['freqdset'] # NPIPE or DX12
    freqfilter = pa["freqfilter"]
    specfilter = pa["specfilter"]

    indir_path = cf[mch]['indir']

    tmask_path = cf[mch][mskset]['tmask']["path"]
    pmask_path = cf[mch][mskset]['pmask']["path"]
    freq_path = cf[mch][freqdset]['path']

    tmask_filename = cf[mch][mskset]['tmask']['filename']
    pmask_filename = cf[mch][mskset]['pmask']['filename']
    freq_filename = cf[mch][freqdset]['filename']

    nside = cf['pa']["nside"]

    def _read(mask_path, mask_filename):
        return hp.read_map(
            '{path}{mask_path}{mask_filename}'
            .format(
                path = indir_path,
                mask_path = tmask_path,
                mask_filename = tmask_filename), dtype=np.bool)
    def _multi(a,b):
        return a*b

    ### Build paths and filenames from config information
    mappath = {
        FREQ:'{path}{freq_path}{freq_filename}'
            .format(
                path = indir_path,
                freq_path = freq_path,
                freq_filename = freq_filename
                    .replace("{freq}", FREQ)
                    .replace("{LorH}", Planckr.LFI.value if int(FREQ)<100 else Planckr.HFI.value)
                    .replace("{nside}", str(nside[0]) if int(FREQ)<100 else str(nside[1]))
                    .replace("{split}", cf['pa']["freqdatsplit"] if "split" in cf[mch][freqdset] else "")
                )
            for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter}

    if tmask_filename is None:
        tmask = None
        tmask_d = None
    else:
        tmask = _read(tmask_path, tmask_filename)
        tmask_d = hp.pixelfunc.ud_grade(tmask, nside_out=nside[0])

    if pmask_filename is None:
        pmask = None
        pmask_d = None
    else:
        pmasks = [_read(pmask_path, a) for a in pmask_filename]
        pmask = functools.reduce(_multi, pmasks)
        pmask_d = hp.pixelfunc.ud_grade(pmask, nside_out=nside[0])


    ## Decide which maps to load 
    flag = False
    for spec in PLANCKSPECTRUM:
        if spec not in specfilter and ("T" in spec):
            flag = True
            break
    if flag and (tmask is not None):
        tmap = {
            FREQ: {
                "map": hp.read_map(mappath[FREQ], field=0),
                "mask": tmask_d if int(FREQ)<100 else tmask
                }for FREQ in PLANCKMAPFREQ
                    if FREQ not in freqfilter
        }
    elif flag and (tmask is None):
        print("Temperature spectra requested, but no Temperature mask provided. Spectra including temperature are skipped")
        tmap = None
    elif flag == False:
        tmap = None

    qmap = {
        FREQ: {
            "map": hp.read_map(mappath[FREQ], field=1),
            "mask": pmask_d if int(FREQ)<100 else pmask
            }for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
    }
    umap = {
        FREQ: {
            "map": hp.read_map(mappath[FREQ], field=2),
            "mask": pmask_d if int(FREQ)<100 else pmask
            }for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
    }
    return [tmap, qmap, umap]


@log_on_start(INFO, "Trying to load spectrum {filename}")
@log_on_end(DEBUG, "{result} loaded")
def load_weights(path: str, filename: str = 'default') -> Dict[str, Dict]:
    fending = ".npy"
    if os.path.isfile(path+"weights/"+filename+fending):
        data = np.load(path+"weights/"+filename+fending, allow_pickle=True)
        return data.item()
    else:
        print("no existing weights at {}".format(path+"weights/"+filename))
        return None


@log_on_start(INFO, "Trying to load spectrum {fname}")
@log_on_end(DEBUG, "{result} loaded")
def load_synmap(path: str, ddesc: str, dpath: str, fname: str = 'default') -> Dict[str, Dict]:
    fending = ".npy"
    tot_path = path+"map/"+dpath+ddesc+fname+fending
    if os.path.isfile(tot_path):
        data = np.load(tot_path, allow_pickle=True)
        return data
    else:
        print("no existing map at {}".format(tot_path))
        return None


@log_on_start(INFO, "Trying to load spectrum {filename}")
@log_on_end(DEBUG, "{result} loaded")
def load_spectrum(path: str, filename: str = 'default') -> Dict[str, Dict]:
    fending = ".npy"
    if os.path.isfile(path+"spectrum/"+filename+fending):
        data = np.load(path+"spectrum/"+filename+fending, allow_pickle=True)
        return data.item()
    else:
        print("no existing spectrum at {}".format(path+"spectrum/"+filename))
        return None
        

@log_on_start(INFO, "Starting to grab data from frequency channels {freqcomb}")
@log_on_end(DEBUG, "Beamfunction(s) loaded successfully: '{result}' ")
def load_beamf(freqcomb: List, abs_path: str = "") -> Dict:
    """Collects planck beamfunctions (.fits files) and stores to dictionaries. beamf files must be placed in `PATH/beamf/`.

    Args:
        cf (Dict): Configuration as coming from conf.json
        mch (str): ID of the machine the code is executed. Depending on the ID, a different set of configurations is used.
        freqcomb (List): Frequency channels which are to be ignored

    Returns:
        Dict: Planck beamfunctions
    """
    beamf = dict()
    for freqc in freqcomb:
        freqs = freqc.split('-')
        if int(freqs[0]) >= 100 and int(freqs[1]) >= 100:
            beamf.update({
                freqc: {
                    "HFI": fits.open(
                        "{abs_path}{indir_path}{bf_path}{bf_filename}"
                        .format(
                            abs_path = abs_path,
                            indir_path = cf[mch]['indir'],
                            bf_path = cf[mch]["beamf"]['path'],
                            bf_filename = cf[mch]["beamf"]['filename']
                                .replace("{freq1}", freqs[0])
                                .replace("{freq2}", freqs[1])
                        ))
                    }
                })
        elif int(freqs[0]) < 100 and int(freqs[1]) >= 100:
            beamf.update({
                freqc: {
                    "HFI": fits.open(
                        "{abs_path}{indir_path}{bf_path}{bf_filename}"
                        .format(
                            abs_path = abs_path,
                            indir_path = cf[mch]['indir'],
                            bf_path = cf[mch]["beamf"]['path'],
                            bf_filename = cf[mch]["beamf"]['filename']
                                .replace("{freq1}", freqs[1])
                                .replace("{freq2}", freqs[1])
                    ))
                }
            })
            beamf[freqc].update({
                "LFI": fits.open("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/beamf/BeamWF_LFI/LFI_RIMO_R3.31.fits")
            })
        if int(freqs[0]) < 100 and int(freqs[1]) < 100:
             beamf.update({
                freqc: {
                    "LFI": fits.open("/mnt/c/Users/sebas/OneDrive/Desktop/Uni/project/component_separation/data/beamf/BeamWF_LFI/LFI_RIMO_R3.31.fits")
                }})
    return beamf
# %%
def save_map(data: Dict[str, Dict], path: str, filename: str = 'default'):
    
    if os.path.exists(path+filename):
        os.remove(path+filename)
    np.save(path+"map/"+filename, data)


def save_mapfigure(mp, outdir_root, rel_dir, pfiledesc, fname):
    fending = ".jpg"
    mp.savefig('{outdir_root}vis/map/{rel_dir}{pfiledesc}-{fname}{fending}'.format(
        outdir_root = outdir_root,
        rel_dir = rel_dir,
        fname = fname,
        pfiledesc = pfiledesc,
        fending = fending))
    plt.close()

def save_weights(data: Dict[str, Dict], path: str, filename: str = 'default'):
    if os.path.exists(path+filename):
        os.remove(path+filename)
    np.save(path+"weights/"+filename, data)


@log_on_start(INFO, "Saving spectrum to {filename}")
@log_on_end(DEBUG, "Spectrum saved successfully to{filename}")
def save_spectrum(data: Dict[str, Dict], path: str, filename: str = 'default'):
    if os.path.exists(path+filename):
        os.remove(path+filename)
    np.save(path+"spectrum/"+filename, data)