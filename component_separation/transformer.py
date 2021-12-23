import healpy as hp
import numpy as np

from logdecorator import log_on_end, log_on_error, log_on_start
from logging import DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple

import component_separation.MSC.MSC.pospace as ps

    
def alm_s2map(tlm, elm, blm, nsi):

    return hp.alm2map([tlm, elm, blm], nsi)


def map2alm_spin(maps, pmask, spin, lmax):

    return ps.map2alm_spin([maps[1] * pmask, maps[2] * pmask], spin,
                                lmax=lmax)


def map2alm(maps, tmask, lmax):
    assert 0, 'Check the return statement'
    return ps.map2alm([maps[0] * tmask, maps[2] * tmask],
                                lmax=lmax)
                                

@log_on_start(INFO, "Starting to calculate powerspectra.")
@log_on_end(DEBUG, "Spectrum calculated successfully: '{result}' ")
def map2cls(maps, tmask, pmask, powerspectrum_type, lmax, freqcomb, nside_out=[1024,2048], lmax_mask=None):
    """Root function. Forwards request to correct powerspectrum calculator

    Args:
        maps ([type]): [description]
        tmask ([type]): [description]
        pmask ([type]): [description]
        powerspectrum_type ([type]): [description]
        lmax ([type]): [description]
        freqcomb ([type]): [description]
        nside_out (list, optional): [description]. Defaults to [1024,2048].
        lmax_mask ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    assert powerspectrum_type in ['JC', 'pseudo']

    if powerspectrum_type == 'JC':
        Cl_usc = _map2cls(maps, tmask, pmask, lmax, lmax_mask, freqcomb, nside_out)
    elif powerspectrum_type == 'pseudo':
        Cl_usc = _map2pcls(maps, tmask, pmask, lmax, freqcomb, nside_out)

    return Cl_usc


@log_on_start(INFO, "Starting to calculate JC powerspectra.")
@log_on_end(DEBUG, "Spectrum calculated successfully: '{result}' ")
def _map2cls(iqumap, tmask: List, pmask: List, lmax, lmax_mask, freqcomb, nside_out) -> np.array:
    """map 2 powerspectrum, temp and pol, using MSC.pospace

    Args:
        iqumap ([type]): [description]
        tmask (List): [description]
        pmask (List): [description]
        lmax ([type]): [description]
        lmax_mask ([type]): [description]
        freqcomb ([type]): [description]
        nside_out ([type]): [description]

    Returns:
        np.array: [description]
    """
    retval = np.array([
        ps.map2cls(
            tqumap=iqumap[FREQC.split("-")[0]],
            tmask=_ud_grade(tmask[FREQC.split("-")[0]], FREQC.split("-")[0], nside_out),
            pmask=_ud_grade(pmask[FREQC.split("-")[0]], FREQC.split("-")[0], nside_out),
            lmax=lmax,
            lmax_mask=lmax_mask,
            tqumap2=iqumap[FREQC.split("-")[1]],
            tmask2=_ud_grade(tmask[FREQC.split("-")[0]], FREQC.split("-")[1], nside_out),
            pmask2=_ud_grade(pmask[FREQC.split("-")[0]], FREQC.split("-")[1], nside_out)
        ) for FREQC in freqcomb 
    ])

    return retval


@log_on_start(INFO, "Starting to calculate JC powerspectra.")
@log_on_end(DEBUG, "Spectrum calculated successfully: '{result}' ")
def map2cl_ss(qumap, pmask: List, spin, lmax, lmax_mask) -> np.array:
    """map2 powerspectrum _ spin only, single map

    Args:
        qumap ([type]): [description]
        pmask (np.array): [description]
        spin ([type]): [description]
        lmax ([type]): [description]
        lmax_mask ([type]): [description]

    Returns:
        np.array: [description]
    """
    retval = np.array([
        ps.map2cl_spin(qumap=qumap, spin=spin, mask=pmask, lmax=lmax-1, lmax_mask=lmax_mask)
    ])

    return retval


@log_on_start(INFO, "Starting to calculate pseudo-powerspectra")
@log_on_end(DEBUG, "Spectrum calculated successfully: '{result}' ")
def _map2pcls(iqumap, tmask: List, pmask: List, lmax, freqcomb, nside_out) -> np.array:
    """Calculate powerspectrum using healpy and iqumaps

    Args:
        iqumap List[List]: Maps
        tmask (List): [description]
        pmask (List): [description]
        lmax ([type]): [description]
        freqcomb ([type]): [description]
        nside_out ([type]): [description]

    Returns:
        np.array: Powerspectra as provided from hp.anafast
    """
    #TODO Perhaps switch back to masked arrays
    def _maIQU(FREQC, splitID):
        freq1 = FREQC.split("-")[splitID]
        freq2 = FREQC.split("-")[int(not(splitID))]

        if (int(freq1) < 100 and int(freq2) >= 100) or (int(freq1) >= 100 and int(freq2) < 100): #if LFI-HFI, force Nside to nside_out[0]
            ma_map_I = np.ma.masked_array(_ud_grade(iqumap[freq1][0], 0, nside_out), mask=_ud_grade(tmask[freq1], 0, nside_out), fill_value=0)
            ma_map_Q = np.ma.masked_array(_ud_grade(iqumap[freq1][1], 0, nside_out), mask=_ud_grade(pmask[freq1], 0, nside_out), fill_value=0)
            ma_map_U = np.ma.masked_array(_ud_grade(iqumap[freq1][2], 0, nside_out), mask=_ud_grade(pmask[freq1], 0, nside_out), fill_value=0)  
        else:
            ma_map_I = np.ma.masked_array(iqumap[freq1][0], mask=_ud_grade(tmask[freq1], freq1, nside_out), fill_value=0)
            ma_map_Q = np.ma.masked_array(iqumap[freq1][1], mask=_ud_grade(pmask[freq1], freq1, nside_out), fill_value=0)
            ma_map_U = np.ma.masked_array(iqumap[freq1][2], mask=_ud_grade(pmask[freq1], freq1, nside_out), fill_value=0)

        return np.array([ma_map_I.filled(), ma_map_Q.filled(), ma_map_U.filled()])


    def _IQU(FREQC, splitID):
        freq1 = FREQC.split("-")[splitID]
        freq2 = FREQC.split("-")[int(not(splitID))]

        ## Special case, when combined powerspectrum is calculated. For now, assume combined map has Nside=2048
        if freq1 == 'combined' and freq2 == 'combined':
            log_freq1, log_freq2 = '100', '100'
        else:
            log_freq1, log_freq2 = freq1, freq2

        if (int(log_freq1) < 100 and int(log_freq2) >= 100) or (int(log_freq1) >= 100 and int(log_freq2) < 100): #if LFI-HFI, force Nside to nside_out[0]
            map_I = _ud_grade(iqumap[freq1][0], 0, nside_out) * _ud_grade(tmask[freq1], 0, nside_out)
            map_Q = _ud_grade(iqumap[freq1][1], 0, nside_out) * _ud_grade(pmask[freq1], 0, nside_out)
            map_U = _ud_grade(iqumap[freq1][2], 0, nside_out) * _ud_grade(pmask[freq1], 0, nside_out)
        else:
            map_I = iqumap[freq1][0] * _ud_grade(tmask[freq1], freq1, nside_out)
            map_Q = iqumap[freq1][1] * _ud_grade(pmask[freq1], freq1, nside_out)
            map_U = iqumap[freq1][2] * _ud_grade(pmask[freq1], freq1, nside_out)

        return np.array([map_I, map_Q, map_U])

    retval = np.array([
        hp.anafast(
            map1=_IQU(FREQC, 0),
            map2=_IQU(FREQC, 1),
            lmax=lmax
        ) for FREQC in freqcomb 
    ])

    return retval


def _ud_grade(data, FREQ, nside_out):

    ## Special case, when combined powerspectrum is calculated. For now, assume combined map has Nside=2048
    if FREQ == 'combined':
        FREQ = '100'
    if int(FREQ)<100:
        return hp.pixelfunc.ud_grade(data, nside_out=nside_out[0])
    else:
        return hp.pixelfunc.ud_grade(data, nside_out=nside_out[1])