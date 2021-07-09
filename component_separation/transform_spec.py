import logging
#%%

from logging import DEBUG, ERROR, INFO
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np
import functools
import component_separation.io as io
import healpy as hp
from logdecorator import log_on_end, log_on_error, log_on_start
from scipy.signal import savgol_filter
import component_separation.spherelib.python.spherelib.astro as slhpastro


def process_all(data, cf, freqcomb, speccomb, scale, beamf, nside, spectrum_scale, smoothing_window, max_polynom):
    """
    Root function. Executes all tranformations
    """
    if smoothing_window > 0 or max_polynom > 0:
        data = apply_smoothing(data, smoothing_window=smoothing_window, max_polynom=max_polynom)
    data = apply_pixwin(data, freqcomb, nside)
    data = apply_scale(data, spectrum_scale)
    data = apply_beamfunction(data, cf, freqcomb, speccomb, beamf)
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
    """
    # for key, val in data.items():
    #     for k, v in val.items():
    #         data[key][k] = savgol_filter(v, smoothing_window, max_polynom)
    # return data
    #TODO this needs being tested
    return savgol_filter(data, smoothing_window, max_polynom)


@log_on_start(INFO, "Starting to apply Beamfunction")
@log_on_end(DEBUG, "Beamfunction applied successfully: '{result}' ")
def apply_beamfunction(data: np.array, cf, freqcomb, speccomb, beamf: Dict) -> Dict:
    """divides the spectrum derived from channel `ij` and provided via `df_(ij)`,
    by `beamf_i beamf_j` as described by the planck beamf .fits-file header.

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns. works for both, npipe and
        dx12 beams
        beamf: Dict

    Returns:
        np.array: powerspectra including effect of Beam, with spectrum and frequency-combinations in the columns

    """
    TEB_dict = {
        "T": 0,
        "E": 1,
        "B": 2
    }
    LFI_dict = {
        "030": 28,
        "044": 29,
        "070": 30
    }
    lmaxp1 = data.shape[-1]

    for fidx, freql in enumerate(data):
        freqc = freqcomb[fidx].split('-')
        hdul = beamf[freqcomb[fidx]]
        for sidx, specl in enumerate(freql):
            specID = speccomb[sidx]
            if cf['pa']['freqdset'].startswith('DX12'):
                if int(freqc[0]) >= 100 and int(freqc[1]) >= 100:
                    data[fidx, sidx] /= hdul["HFI"][1].data.field(TEB_dict[specID[0]])[:lmaxp1]
                    data[fidx, sidx] /= hdul["HFI"][1].data.field(TEB_dict[specID[1]])[:lmaxp1]
                elif int(freqc[0]) < 100 and int(freqc[1]) < 100:
                    for freq in freqc:
                        b = np.sqrt(hdul["LFI"][LFI_dict[freq]].data.field(0))
                        buff = np.concatenate((
                            b[:min(lmaxp1, len(b))],
                            np.array([np.NaN for n in range(max(0, lmaxp1-len(b)))])))
                        data[fidx, sidx] /= buff
                else:
                    b = np.sqrt(hdul["LFI"][LFI_dict[freqc[0]]].data.field(0))
                    buff = np.concatenate((
                        b[:min(lmaxp1, len(b))],
                        np.array([np.NaN for n in range(max(0, lmaxp1-len(b)))])))
                    data[fidx, sidx] /= buff
                    data[fidx, sidx] /= hdul["HFI"][1].data.field(TEB_dict[specID[1]])[:lmaxp1]

            elif cf['pa']['freqdset'].startswith('NPIPE'):
                    ### now that all cross beamfunctions exist, and beamf
                    ### files have the same structure, no difference between applying lfis and hfis anymore
                    data[fidx, sidx] /= hdul["HFI"][1].data.field(TEB_dict[specID[0]])[:lmaxp1]
                    data[fidx, sidx] /= hdul["HFI"][1].data.field(TEB_dict[specID[1]])[:lmaxp1]
            else:
                print("Error checking data for applying beamfunction to dataset. Dataset might not be supported. Exiting..")
                sys.exit()
    return data