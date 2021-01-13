"""
io.py: Filehandling functions

"""

__author__ = "S. Belkner"


from astropy.io import fits
from logdecorator import log_on_end, log_on_error, log_on_start
from logging import DEBUG, ERROR, INFO
import os
import sys
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
def get_data(path: str, freqfilter: List[str]) -> List[Dict]:
    """Collects planck maps (.fits files) and stores to dictionaries. Mask data must be placed in `PATH/mask/`,
    Map data in `PATH/map/`.
    Args:
        path (str): relative path to root dir of the data. Must end with '/'
        freqfilter (List[str]): Frequency channels which are to be ignored

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
                nside = PLANCKMAPNSIDE[0] if int(FREQ)<100 else PLANCKMAPNSIDE[1])
            for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter}
    tmask = hp.read_map('{}/mask/HFI_Mask_GalPlane-apo0_2048_R2.00.fits'.format(path), field=2, dtype=np.float64)
    tmask_d = hp.pixelfunc.ud_grade(tmask, nside_out=1024)

    hp_psmask = hp.read_map('{}/mask/psmaskP_2048.fits.gz'.format(path), dtype=np.bool)
    hp_gmask = hp.read_map('{}/mask/gmaskP_apodized_0_2048.fits.gz'.format(path), dtype=np.bool)
    pmask = hp_psmask*hp_gmask
    pmask_d = hp.pixelfunc.ud_grade(pmask, nside_out=1024)

    tmap = {
        FREQ: {
            "header": {
                "nside" : PLANCKMAPNSIDE[0] if int(FREQ)<100 else PLANCKMAPNSIDE[1],
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
                "nside" : PLANCKMAPNSIDE[0] if int(FREQ)<100 else PLANCKMAPNSIDE[1],
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
                "nside" : PLANCKMAPNSIDE[0] if int(FREQ)<100 else PLANCKMAPNSIDE[1],
                "freq" : FREQ,
                "LorH" : Planckr.LFI if int(FREQ)<100 else Planckr.HFI
            },
            "map": hp.read_map(mappath[FREQ], field=2),
            "mask": pmask_d if int(FREQ)<100 else pmask
            }for FREQ in PLANCKMAPFREQ
                if FREQ not in freqfilter
    }
    return [tmap, qmap, umap]


def get_beamf(freqcomb: List) -> Dict:
    beamf = dict()
    for fkey in freqcomb:
        freqs = fkey.split('-')
        beamf.update({fkey: fits.open("data/beamf/BeamWf_HFI_R3.01/Bl_TEB_R3.01_fullsky_{}x{}.fits".format(*freqs))})
    return beamf
# %%
