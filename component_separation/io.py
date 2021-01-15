"""
io.py: Filehandling functions

"""

__author__ = "S. Belkner"


from astropy.io import fits
from logdecorator import log_on_end, log_on_error, log_on_start
from logging import DEBUG, ERROR, INFO
import os
import sys
import functools
import os.path
import numpy as np
from typing import Dict, List, Optional, Tuple
from component_separation.cs_util import Planckf, Plancks, Planckr
import healpy as hp
PLANCKMAPFREQ = [p.value for p in list(Planckf)]
PLANCKMAPNSIDE = [1024, 2048]
PLANCKSPECTRUM = [p.value for p in list(Plancks)]

#%% Collect maps
@log_on_start(INFO, "Starting to grab data without {freqfilter}")
@log_on_end(DEBUG, "Data without '{freqfilter}' loaded successfully: '{result}' ")
def get_data(path: str, freqfilter: List[str], tmask_filename: str, pmask_filename: List[str], nside: List[int] = PLANCKMAPNSIDE) -> List[Dict]:
    """Collects planck maps (.fits files) and stores to dictionaries. Mask data must be placed in `PATH/mask/`,
    Map data in `PATH/map/`.
    Args:
        path (str): Relative path to root dir of the data. Must end with '/'
        freqfilter (List[str]): Frequency channels which are to be ignored
        tmask_filename (str): name of the mask for intensity maps
        pmask_filename (List[str]): list of names of the masks for polarisation maps. They will be reduced to one mask

    Returns:
        List[Dict]: Planck maps (data and masks) and some header information

    Doctest:
    >>> get_data(freqfilter=freqfilter) 
    NotSureWhatToExpect
    """    
    mappath = {
        FREQ:'{path}map/frequency/{LorH}_SkyMap_{freq}-field_{nside}_R3.00_full.fits'
            .format(
                path = path,
                LorH = Planckr.LFI.value if int(FREQ)<100 else Planckr.HFI.value,
                freq = FREQ,
                nside = nside[0] if int(FREQ)<100 else nside[1])
            for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter}

    tmask = hp.read_map('{}mask/{}'.format(path, tmask_filename), field=2, dtype=np.float64)
    tmask_d = hp.pixelfunc.ud_grade(tmask, nside_out=nside[0])

    def multi(a,b):
        return a*b
    pmasks = [hp.read_map('{}mask/{}'.format(path, a), dtype=np.bool) for a in pmask_filename]
    pmask = functools.reduce(multi, pmasks)
    pmask_d = hp.pixelfunc.ud_grade(pmask, nside_out=nside[0])

    print(mappath)
    tmap = {
        FREQ: {
            "header": {
                "nside" : nside[0] if int(FREQ)<100 else nside[1],
                "freq" : FREQ,
                "LorH" : Planckr.LFI if int(FREQ)<100 else Planckr.HFI
            },
            "map": hp.read_map(mappath[FREQ], field=0),
            "mask": tmask_d if int(FREQ)<100 else tmask
            }for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
    }

    qmap = {
        FREQ: {
            "header": {
                "nside" : nside[0] if int(FREQ)<100 else nside[1],
                "freq" : FREQ,
                "LorH" : Planckr.LFI if int(FREQ)<100 else Planckr.HFI
            },
            "map": hp.read_map(mappath[FREQ], field=1),
            "mask": pmask_d if int(FREQ)<100 else pmask
            }for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
    }

    umap = {
        FREQ: {
            "header": {
                "nside" : nside[0] if int(FREQ)<100 else nside[1],
                "freq" : FREQ,
                "LorH" : Planckr.LFI if int(FREQ)<100 else Planckr.HFI
            },
            "map": hp.read_map(mappath[FREQ], field=2),
            "mask": pmask_d if int(FREQ)<100 else pmask
            }for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
    }
    return [tmap, qmap, umap]

def save_spectrum(data: Dict[str, Dict], path: str, filename: str = 'default.npy'):
    np.save(path+filename, data)

def load_spectrum(path: str, filename: str = 'default.npy') -> Dict[str, Dict]:
    if os.path.isfile(path+filename):
        data = np.load(path+filename, allow_pickle=True)
        return data.item()
    else:
        return None

@log_on_start(INFO, "Starting to grab data from frequency channels {freqcomb}")
@log_on_end(DEBUG, "Beamfunction(s) loaded successfully: '{result}' ")
def get_beamf(path: str, freqcomb: List) -> Dict:
    """Collects planck beamfunctions (.fits files) and stores to dictionaries. beamf files must be placed in `PATH/beamf/`.

    Args:
        path (str): Relative path to root dir of the data. Must end with '/'
        freqcomb (List): Frequency channels which are to be ignored

    Returns:
        Dict: Planck beamfunctions
    """    
    beamf = dict()
    for fkey in freqcomb:
        freqs = fkey.split('-')
        beamf.update({fkey: fits.open("{}beamf/BeamWf_HFI_R3.01/Bl_TEB_R3.01_fullsky_{}x{}.fits".format(path, *freqs))})
    return beamf
# %%
