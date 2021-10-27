import logging
from logging import DEBUG, ERROR, INFO

import healpy as hp
import numpy as np
from logdecorator import log_on_end, log_on_error, log_on_start

from component_separation.cs_util import Helperfunctions as hpf


@log_on_start(INFO, "Starting to process spectrum")
@log_on_end(DEBUG, "Spectrum processed successfully: '{result}' ")
def process_all(data, freqcomb, beamf, nside, spectrum_scale):
    """
    Root function. Executes all transformations
    """
    data = apply_pixwin(data, freqcomb, nside)
    data = apply_scale(data, spectrum_scale)
    data = apply_beamfunction(data, beamf, freqcomb)
    return data


@log_on_start(INFO, "Starting to apply pixwindow onto data {data}")
@log_on_end(DEBUG, "Pixwindow scaled successfully: '{result}' ")
def apply_pixwin(data: np.array, freqcomb, nside) -> np.array:
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
def apply_scale(data: np.array, scale: str = 'C_l') -> np.array:
    """
    Assumes C_l as input. output may be D_l. Guaranteed to multiply spectrum by 1e12.
    """
    lmaxp1 = data.shape[-1]
    if scale == "D_l":
        data *= np.array([hpf.llp1e12(idx) for idx in range(lmaxp1)])
    elif scale == "C_l":
        data *= np.array([1e12 for idx in range(lmaxp1)])
    return data


@log_on_start(INFO, "Starting to apply Beamfunction")
@log_on_end(DEBUG, "Beamfunction applied successfully: '{result}' ")
def apply_beamfunction(data: np.array, beamf, freqcomb) -> np.array:
    """divides the spectrum derived from channel `ij` and provided via `df_(ij)`,
    by `beamf_i beamf_j` as described by the planck beamf .fits-file header.

    Args:
        data (np.array): powerspectra with spectrum and frequency-combinations in the columns. works for both, npipe and
        dx12 beams
        beamf: Dict

    Returns:
        np.array: powerspectra including effect of Beam, with spectrum and frequency-combinations in the columns

    """
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


    #TODO perhaps make smica fit with LFI and HFI up to ell = 1000.
def cov2cov_smooth(cov, cutoff):
    """
    currently takes any LFI[0] X (LFI or HFI) crossspectra and applies a windowfunction, effectively setting it to zero.
    This needs to be done as the freq 030 crosspowerspectra have unphysical behaviour for ell>900. This is also true for
    other LFIs. BUT: if all are set to zero, we loose cmb-signal-power in EE, as there is still signal on that scale.

    Thus we need to cherry pick which crossspectra can actually be set to zero without impacting the SMICA fit and MV weights..
    """
    for spec in range(cov.shape[0]):
        for n in range(1):
            for m in range(cov.shape[2]):
                if n != m:
                    cov[spec,n,m,cutoff:] = 0#np.zeros_like(cov[spec,n,m,cutoff:])
                    cov[spec,m,n,cutoff:] = 0#np.zeros_like(cov[spec,m,n,cutoff:])

    for spec in range(cov.shape[0]):
        n=1
        for m in range(cov.shape[2]):
            if m>n:
                cov[spec,n,m,1500:] = 0#np.zeros_like(cov[spec,n,m,cutoff:])
                cov[spec,m,n,1500:] = 0#np.zeros_like(cov[spec,m,n,cutoff:])

    for spec in range(cov.shape[0]):
        n=2
        for m in range(cov.shape[2]):
            if m>n:
                cov[spec,n,m,1500:] = 0#np.zeros_like(cov[spec,n,m,cutoff:])
                cov[spec,m,n,1500:] = 0#np.zeros_like(cov[spec,m,n,cutoff:])
    return cov


@hpf.deprecated
@log_on_start(INFO, "Starting to calculate channel weights with covariances {cov}")
@log_on_end(DEBUG, "channel weights calculated successfully: '{result}' ")
def calculate_weights(cov: np.array, freqs, Tscale: str = r"K_CMB") -> np.array:
    """Calculates weightings of the respective Frequency channels
    Args:
        cov (Dict): The inverted covariance matrices of Dimension [lmax,Nspec,Nspec]

    Returns:
        np.array: The weightings of the respective Frequency channels
    """

    elaw = np.array([trsf_m.tcmb2trj_sc(FREQ, fr=r'K_CMB', to=Tscale) for FREQ in freqs])
    weight_arr = np.zeros(shape=(np.take(cov.shape, [0,1,-1])))
    for spec in range(weight_arr.shape[0]):
        for l in range(cov.shape[-1]):
            weight_arr[spec,:,l] = np.array([
                    (cov[spec,:,:,l] @ elaw) / (elaw.T @ cov[spec,:,:,l] @ elaw)])
    return weight_arr