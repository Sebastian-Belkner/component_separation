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
import component_separation.spherelib.python.spherelib.astro as slhpastro


def process_all(data, cf, scale, beamf, nside, spectrum_scale, smoothing_window, max_polynom):
    """
    Root function. Executes all tranformations
    """
    data_prep = data
    if smoothing_window > 0 or max_polynom > 0:
        data_prep = apply_smoothing(data_prep, smoothing_window=smoothing_window, max_polynom=max_polynom)
    data_prep = apply_pixwin(data_prep, nside)
    data_prep = apply_scale(data_prep, spectrum_scale)
    data_prep = apply_beamfunction(data_prep, cf, beamf)
    return data_prep


@log_on_start(INFO, "Starting to apply pixwindow onto data {data}")
@log_on_end(DEBUG, "Data scaled successfully: '{result}' ")
def apply_pixwin(data: Dict, nside) -> Dict:
    """Applies Pixel Windowfunction with nside specified

    Args:
        data (Dict): powerspectra with spectrum and frequency-combinations in the columns
        
    Returns:
        Dict: Pixel Windowfunction applied Powerspectra with spectrum and frequency-combinations in the columns
    """
    for freqc, spec in data.items():
        for specID, val in spec.items():
            lmax = len(next(iter((next(iter(data.values()))).values())))
            if int(freqc.split("-")[0]) < 100:
                data[freqc][specID] /= hp.pixwin(nside[0])[:lmax]
            else:
                data[freqc][specID] /= hp.pixwin(nside[1])[:lmax]
            if int(freqc.split("-")[1]) < 100:
                data[freqc][specID] /= hp.pixwin(nside[0])[:lmax]
            else:
                data[freqc][specID] /= hp.pixwin(nside[1])[:lmax]
    return data


@log_on_start(INFO, "Starting to apply scaling onto data {data}")
@log_on_end(DEBUG, "Data scaled successfully: '{result}' ")
def apply_scale(data: Dict, scale: str = 'C_l') -> Dict:
    """
    Assumes C_l as input. output may be D_l. Guaranteed to multiply spectrum by 1e12.
    """
    for freqc, spec in data.items():
        for specID, val in spec.items():
            lmax = len(next(iter((next(iter(data.values()))).values())))
            if scale == "D_l":
                sc = np.array([hpf.llp1e12(idx) for idx in range(lmax)])
            elif scale == "C_l":
                sc = np.array([1e12 for idx in range(lmax)])
            data[freqc][specID] *= sc
    return data


def apply_smoothing(data, smoothing_window=5, max_polynom=2):
    """
    smoothes the powerspectrum using a savgol_filter. Possibly needed for some cross-powerspectra (LFI)
    """
    from scipy.signal import savgol_filter
    for key, val in data.items():
        for k, v in val.items():
            data[key][k] = savgol_filter(v, smoothing_window, max_polynom)
    return data


@log_on_start(INFO, "Starting to apply Beamfunction")
@log_on_end(DEBUG, "Beamfunction applied successfully: '{result}' ")
def apply_beamfunction(data: Dict,  cf, beamf: Dict) -> Dict:
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
    if cf['pa']['freqdset'].startswith('DX12'):
        for freqc, spec in data.items():
            freqs = freqc.split('-')
            hdul = beamf[freqc]
            for specID, val in spec.items():
                lmaxp1 = len(data[freqc][specID])
                if int(freqs[0]) >= 100 and int(freqs[1]) >= 100:
                    data[freqc][specID] /= hdul["HFI"][1].data.field(TEB_dict[specID[0]])[:lmaxp1]
                    data[freqc][specID] /= hdul["HFI"][1].data.field(TEB_dict[specID[1]])[:lmaxp1]
                elif int(freqs[0]) < 100 and int(freqs[1]) < 100:
                    for freq in freqs:
                        b = np.sqrt(hdul["LFI"][LFI_dict[freq]].data.field(0))
                        buff = np.concatenate((
                            b[:min(lmaxp1, len(b))],
                            np.array([np.NaN for n in range(max(0, lmaxp1-len(b)))])))
                        data[freqc][specID] /= buff
                else:
                    b = np.sqrt(hdul["LFI"][LFI_dict[freqs[0]]].data.field(0))
                    buff = np.concatenate((
                        b[:min(lmaxp1, len(b))],
                        np.array([np.NaN for n in range(max(0, lmaxp1-len(b)))])))
                    data[freqc][specID] /= buff
                    data[freqc][specID] /= hdul["HFI"][1].data.field(TEB_dict[specID[1]])[:lmaxp1]

    elif cf['pa']['freqdset'].startswith('NPIPE'):
    	### now that all cross beamfunctions exist, and beamf
        ### files have the same structure, no difference between applying lfis and hfis anymore
        for freqc, spec in data.items():
            freqs = freqc.split('-')
            hdul = beamf[freqc]
            for specID, val in spec.items():
                lmaxp1 = len(data[freqc][specID])
                data[freqc][specID] /= hdul["HFI"][1].data.field(TEB_dict[specID[0]])[:lmaxp1]
                data[freqc][specID] /= hdul["HFI"][1].data.field(TEB_dict[specID[1]])[:lmaxp1]
    else:
        print("Error applying beamfunction to dataset. Dataset might not be supported. Exiting..")
        sys.exit()
    return data