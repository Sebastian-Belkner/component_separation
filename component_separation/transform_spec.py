import functools
import logging
from logging import DEBUG, ERROR, INFO
from types import MappingProxyType
from typing import Dict, List, Optional, Tuple

import healpy as hp
import numpy as np
from logdecorator import log_on_end, log_on_error, log_on_start
from scipy.signal import savgol_filter

import component_separation.spherelib.python.spherelib.astro as slhpastro
from component_separation.cs_util import Helperfunctions as hpf

#%%



def process_all(data, freqcomb, beamf, nside, spectrum_scale, smoothing_window, max_polynom):
    """
    Root function. Executes all tranformations
    """
    if smoothing_window > 0 or max_polynom > 0:
        data = apply_smoothing(data, smoothing_window=smoothing_window, max_polynom=max_polynom)
    data = apply_pixwin(data, freqcomb, nside)
    data = apply_scale(data, spectrum_scale)
    data = apply_beamfunction(data, beamf, freqcomb)
    return data


@log_on_start(INFO, "Starting to apply pixwindow onto data {data}")
@log_on_end(DEBUG, "Data scaled successfully: '{result}' ")
def apply_pixwin(data: np.array, freqcomb, nside) -> Dict:
    """Applies Pixel Windowfunction with nside specified
    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        
    Returns:
        Dict: Pixel Windowfunction applied Powerspectra with spectrum and frequency-combinations in the columns
    """
    for fidx, freql in enumerate(data):
        for sidx, specl in enumerate(freql):
            lmaxp1 = data.shape[-1]
            freqc = freqcomb[fidx]
            if int(freqc.split("-")[0]) < 100:
                data[fidx,sidx] /= hp.pixwin(nside[0],lmax=lmaxp1-1)
            else:
                data[fidx,sidx] /= hp.pixwin(nside[1],lmax=lmaxp1-1)
            if int(freqc.split("-")[1]) < 100:
                data[fidx,sidx] /= hp.pixwin(nside[0],lmax=lmaxp1-1)
            else:
                data[fidx,sidx] /= hp.pixwin(nside[1],lmax=lmaxp1-1)
    return data


@log_on_start(INFO, "Starting to apply scaling onto data {data}")
@log_on_end(DEBUG, "Data scaled successfully: '{result}' ")
def apply_scale(data: np.array, scale: str = 'C_l') -> Dict:
    """
    Assumes C_l as input. output may be D_l. Guaranteed to multiply spectrum by 1e12.
    """
    lmaxp1 = data.shape[-1]
    if scale == "D_l":
        data *= np.array([hpf.llp1e12(idx) for idx in range(lmaxp1)])
    elif scale == "C_l":
        data *= np.array([1e12 for idx in range(lmaxp1)])
    return data


def apply_smoothing(data, smoothing_window=5, max_polynom=2):
    """
    smoothes the powerspectrum using a savgol_filter. Possibly needed for some cross-powerspectra (LFI)
    Works for any data input dimension
    """
    #TODO this needs being tested
    buff = data.copy()
    if len(data.shape)==1:
        return savgol_filter(data, smoothing_window, max_polynom)
    elif len(data.shape)==2:
        for n in range(len(data)):
            buff[n] = savgol_filter(data[n], smoothing_window, max_polynom)
        return buff
    elif len(data.shape)==3:
        for n in range(len(data)):
            for m in range(len(data)):
                buff[n,m] = savgol_filter(data[n,m], smoothing_window, max_polynom)
        return buff


@log_on_start(INFO, "Starting to apply Beamfunction")
@log_on_end(DEBUG, "Beamfunction applied successfully: '{result}' ")
def apply_beamfunction(data: np.array, beamf, freqcomb) -> Dict:
    """divides the spectrum derived from channel `ij` and provided via `df_(ij)`,
    by `beamf_i beamf_j` as described by the planck beamf .fits-file header.

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns. works for both, npipe and
        dx12 beams
        beamf: Dict

    Returns:
        np.array: powerspectra including effect of Beam, with spectrum and frequency-combinations in the columns

    """
    lmaxp1 = data.shape[-1]
    freqmapping = []
    specmapping = []
    for fidx, freql in enumerate(data):
        freqc = freqcomb[fidx].split('-')
        if freqc[0] not in freqmapping:
            freqmapping+=[freqc[0]]
        if freqc[1] not in freqmapping:
            freqmapping+=[freqc[1]]
        freqmapping = sorted(freqmapping)
        fida = freqmapping.index(freqc[0])
        fidb = freqmapping.index(freqc[1])
        
        specmapping = {
                0:[0,0],
                1:[1,1],
                2:[2,2],
                3:[0,1],
                4:[0,2],
                5:[1,2],
                6:[1,0],
                7:[2,0],
                8:[2,1],
            }
        
        for sidx, specl in enumerate(freql):
            sida, sidb = specmapping[sidx]
            data[fidx, sidx] /= beamf[sida,fida,fidb]
            data[fidx, sidx] /= beamf[sidb,fida,fidb]

    return data