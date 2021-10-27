import healpy as hp
import numpy as np

from component_separation.cs_util import Config
from component_separation.io import IO
import component_separation.MSC.MSC.pospace as ps

from logdecorator import log_on_end, log_on_error, log_on_start
from logging import DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

csu = Config()
io = IO(csu)


@log_on_start(INFO, "Starting to calculate powerspectra.")
@log_on_end(DEBUG, "Spectrum calculated successfully: '{result}' ")
def map2cls(maps, powerspectrum_type, tmask, pmask):
    """
    Root function. Forwards request to correct powerspectrum calculator
    """
    assert powerspectrum_type in ['Chonetal', 'pseudo']

    if powerspectrum_type == 'Chonetal':
        Cl_usc = _map2cls(maps, tmask, pmask)
    elif powerspectrum_type == 'pseudo':
        Cl_usc = _map2pcls(maps, tmask, pmask)
    return Cl_usc


@log_on_start(INFO, "Starting to calculate Chonetal powerspectra.")
@log_on_end(DEBUG, "Spectrum calculated successfully: '{result}' ")
def _map2cls(iqumap, tmask: List, pmask: List) -> np.array:
    """Calculate powerspectrum using MSC.pospace and iqumaps
    Args:
        iqumap List[Dict[str, Dict]]: Planck maps (data and masks) and some header information
        tmask :
        pmask :

    Returns:
        np.array: Powerspectra as provided from MSC.pospace
    """
    retval = np.array([
        ps.map2cls(
            tqumap=iqumap[FREQC.split("-")[0]],
            tmask=_ud_grade(tmask[FREQC.split("-")[0]], FREQC.split("-")[0]),
            pmask=_ud_grade(pmask[FREQC.split("-")[0]], FREQC.split("-")[0]),
            lmax=csu.lmax,
            lmax_mask=csu.lmax_mask,
            tqumap2=iqumap[FREQC.split("-")[1]],
            tmask2=_ud_grade(tmask[FREQC.split("-")[0]], FREQC.split("-")[1]),
            pmask2=_ud_grade(pmask[FREQC.split("-")[0]], FREQC.split("-")[1])
        ) for FREQC in csu.freqcomb 
    ])
    return retval


@log_on_start(INFO, "Starting to calculate pseudo-powerspectra")
@log_on_end(DEBUG, "Spectrum calculated successfully: '{result}' ")
def _map2pcls(iqumap, tmask: List, pmask: List) -> np.array:
    """Calculate powerspectrum using healpy and iqumaps
    Args:
        iqumap Dict[List]: Planck maps
        tmask :
        pmask :

    Returns:
        np.array: Powerspectra as provided from hp.anafast
    """
    def _maIQU(FREQC, splitID):
        freq1 = FREQC.split("-")[splitID]
        freq2 = FREQC.split("-")[int(not(splitID))]
        if (int(freq1) < 100 and int(freq2) >= 100) or (int(freq1) >= 100 and int(freq2) < 100): #if LFI-HFI, force Nside to nside_out[0]
            ma_map_I = np.ma.masked_array(_ud_grade(iqumap[freq1][0], 0), mask=_ud_grade(~tmask[freq1], 0), fill_value=0)
            ma_map_Q = np.ma.masked_array(_ud_grade(iqumap[freq1][1], 0), mask=_ud_grade(~pmask[freq1], 0), fill_value=0)
            ma_map_U = np.ma.masked_array(_ud_grade(iqumap[freq1][2], 0), mask=_ud_grade(~pmask[freq1], 0), fill_value=0)  
        else:
            ma_map_I = np.ma.masked_array(iqumap[freq1][0], mask=_ud_grade(~tmask[freq1], freq1), fill_value=0)
            ma_map_Q = np.ma.masked_array(iqumap[freq1][1], mask=_ud_grade(~pmask[freq1], freq1), fill_value=0)
            ma_map_U = np.ma.masked_array(iqumap[freq1][2], mask=_ud_grade(~pmask[freq1], freq1), fill_value=0)
        return np.array([ma_map_I.filled(), ma_map_Q.filled(), ma_map_U.filled()])

    retval = np.array([
        hp.anafast(
            map1=_maIQU(FREQC, 0),
            map2=_maIQU(FREQC, 1),
            lmax=4000
        ) for FREQC in csu.freqcomb 
    ])
    return retval


def _ud_grade(data, FREQ):
    if int(FREQ)<100:
        return hp.pixelfunc.ud_grade(data, nside_out=csu.nside_out[0])
    else:
        return hp.pixelfunc.ud_grade(data, nside_out=csu.nside_out[1])